const std = @import("std");
const Allocator = std.mem.Allocator;
const shared = @import("shared_benchmarks.zig");
const HNSW = @import("zvdb").HNSW;

// Helper function to calculate Euclidean distance
fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, 0..) |_, i| {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std.math.sqrt(sum);
}

pub fn randomPointNorm(allocator: std.mem.Allocator, dim: usize) ![]f32 {
    const point = try allocator.alloc(f32, dim);
    for (point) |*v| {
        v.* = std.crypto.random.floatNorm(f32);
    }
    return point;
}

pub const RecallTestResults = struct {
    operation: []const u8,
    num_points: usize,
    dimensions: usize,
    num_queries: ?usize,
    k: usize,
    m: usize,
    ef_c: usize,
    min_recall: f32,
    average_recall: f32,
    max_recall: f32,
    p90_recall: f32, // % of queries that returned recall >= 90%
    num_threads: ?usize,
    total_time_ns: u64,
    insertions_per_second: f64,
    searches_per_second: f64,

    pub fn format(
        self: RecallTestResults,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("{s} Benchmark:\n", .{self.operation});
        try writer.print("  Points: {d}\n", .{self.num_points});
        try writer.print("  Dimensions: {d}\n", .{self.dimensions});
        if (self.num_queries) |queries| {
            try writer.print("  Queries: {d}\n", .{queries});
        }
        try writer.print("  k: {d}\n", .{self.k});
        try writer.print("  m: {d}\n", .{self.m});
        try writer.print("  ef_c: {d}\n", .{self.ef_c});
        try writer.print("  Min Recall: {d:.2}\n", .{self.min_recall});
        try writer.print("  Average Recall: {d:.2}\n", .{self.average_recall});
        try writer.print("  Max Recall: {d:.2}\n", .{self.max_recall});
        try writer.print("  P90 Recall: {d:.2}\n", .{self.p90_recall});
        if (self.num_threads) |threads| {
            try writer.print("  Threads: {d}\n", .{threads});
        }
        try writer.print("  Total time: {d:.2} seconds\n", .{@as(f64, @floatFromInt(self.total_time_ns)) / 1e9});
        try writer.print("  Insertions per second: {d:.2}\n", .{self.insertions_per_second});
        try writer.print("  Searches per second: {d:.2}\n", .{self.searches_per_second});
    }

    pub fn toCsv(self: RecallTestResults) []const u8 {
        return std.fmt.allocPrint(std.heap.page_allocator, "{s},{d},{d},{d},{d},{d},{d},{d:.2},{d:.2},{d:.2},{d:.2},{d},{d},{d:.2},{d:.2}", .{
            self.operation,
            self.num_points,
            self.dimensions,
            self.num_queries orelse 0,
            self.k,
            self.m,
            self.ef_c,
            self.min_recall,
            self.average_recall,
            self.max_recall,
            self.p90_recall,
            self.num_threads orelse 1,
            self.total_time_ns,
            self.insertions_per_second,
            self.searches_per_second,
        }) catch unreachable;
    }
};

pub fn runRecall(allocator: Allocator, num_points: usize, num_queries: usize, dim: usize, m: usize, ef_c: usize, k: usize) !RecallTestResults {
    var hnsw = HNSW(f32).init(allocator, m, ef_c);
    defer hnsw.deinit();

    var recall_sum: f32 = 0;
    var min_recall: f32 = 1.0;
    var max_recall: f32 = 0.0;
    var recalls_over_90: usize = 0;
    var points = try std.ArrayList([]f32).initCapacity(allocator, num_points);
    defer points.deinit();

    const ef_search = ef_c / 4;
    var timer = try std.time.Timer.start();
    const start = timer.lap();
    // Insert points
    for (0..num_points) |_| {
        const point = try randomPointNorm(allocator, dim);
        try hnsw.insert(point);
        try points.append(point);
    }
    const insert_end = timer.read();
    const insertion_elapsed_ns = insert_end - start;
    const insertions_per_second = @as(f64, @floatFromInt(num_points)) / (@as(f64, @floatFromInt(insertion_elapsed_ns)) / 1e9);

    // Search for nearest neighbors
    for (0..num_queries) |_| {
        const query = try randomPointNorm(allocator, dim);
        defer allocator.free(query);

        const point_distances = try allocator.alloc(f32, num_points);
        defer allocator.free(point_distances);
        for (0..num_points) |i| {
            const point = points.items[i];
            point_distances[i] = euclideanDistance(query, point);
        }
        const results = try hnsw.search(query, k, ef_search);
        defer allocator.free(results);

        // Check recall
        const BruteForceContext = struct {
            point_distances: []const f32,
            const Self = @This();

            fn lessThan(context: Self, a: usize, b: usize) std.math.Order {
                const dist_a = context.point_distances[a];
                const dist_b = context.point_distances[b];
                return std.math.order(dist_a, dist_b);
            }
        };
        const context = BruteForceContext{ .point_distances = point_distances };
        var brute_force_pq = std.PriorityQueue(usize, BruteForceContext, BruteForceContext.lessThan).init(allocator, context);
        defer brute_force_pq.deinit();
        for (0..num_points) |i| {
            try brute_force_pq.add(i);
        }

        // According to https://github.com/nmslib/hnswlib/blob/master/TESTING_RECALL.md
        // recall is measured by len(intersection(actual, expected)) / k
        var expected_ids = std.AutoHashMap(usize, void).init(allocator);
        defer expected_ids.deinit();
        for (0..k) |_| {
            const kth_element = brute_force_pq.remove();
            try expected_ids.put(kth_element, {});
        }
        var num_correct: usize = 0;
        for (results) |result| {
            if (expected_ids.contains(result.id)) {
                num_correct += 1;
            }
        }
        const recall = @as(f32, @floatFromInt(num_correct)) / @as(f32, @floatFromInt(k));
        recall_sum += recall;
        min_recall = @min(min_recall, recall);
        max_recall = @max(max_recall, recall);
        if (recall >= 0.9) {
            recalls_over_90 += 1;
        }
    }
    const average_recall = recall_sum / @as(f32, @floatFromInt(num_queries));

    // Clean up
    for (points.items) |point| {
        allocator.free(point);
    }

    const search_end = timer.read();
    const search_elapsed_ns = search_end - insert_end;
    const searches_per_second = @as(f64, @floatFromInt(num_queries)) / (@as(f64, @floatFromInt(search_elapsed_ns)) / 1e9);

    const total_elapsed_ns = search_end - start;
    return RecallTestResults{
        .operation = "Recall",
        .num_points = num_points,
        .dimensions = dim,
        .num_queries = num_queries,
        .k = k,
        .m = m,
        .ef_c = ef_c,
        .min_recall = min_recall,
        .average_recall = average_recall,
        .max_recall = max_recall,
        .p90_recall = @as(f32, @floatFromInt(recalls_over_90)) / @as(f32, @floatFromInt(num_queries)),
        .num_threads = 1,
        .total_time_ns = total_elapsed_ns,
        .insertions_per_second = insertions_per_second,
        .searches_per_second = searches_per_second,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const num_points = 10000;
    const num_queries = 1000;
    const dimensions = &[_]usize{ 32, 64, 128, 512, 768, 1024 };
    const k_values = &[_]usize{ 10, 25, 50, 100 };
    const m_values = &[_]usize{ 16, 32, 48, 64 };
    const ef_c_values = &[_]usize{ 200, 400, 800 }; // use ef_search = ef_c / 4

    var csv_out_file = try std.fs.cwd().createFile("recall_results.csv", .{});
    defer csv_out_file.close();

    try csv_out_file.writeAll("operation,num_points,dimensions,num_queries,k,m,ef_c,min_recall,average_recall,max_recall,p90_recall,num_threads,total_time_ns,insertions_per_second,searches_per_second\n");
    for (ef_c_values) |ef_c| {
        for (k_values) |k| {
            for (dimensions) |dim| {
                for (m_values) |m| {
                    const result = try runRecall(allocator, num_points, num_queries, dim, m, ef_c, k);
                    std.log.info("{}", .{result});
                    try csv_out_file.writeAll(result.toCsv());
                    try csv_out_file.writeAll("\n");
                }
            }
        }
    }
}
