const std = @import("std");
const testing = std.testing;
const HNSW = @import("hnsw.zig").HNSW;

// Helper function to create a normally distrubted random point
fn randomPoint(allocator: std.mem.Allocator, dim: usize) ![]f32 {
    const point = try allocator.alloc(f32, dim);
    for (point) |*v| {
        v.* = std.crypto.random.floatNorm(f32);
    }
    return point;
}

// Helper function to calculate Euclidean distance
fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, 0..) |_, i| {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std.math.sqrt(sum);
}

test "HNSW - Basic Functionality" {
    const ef_construction = 200;
    const ef_search = 50;
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, ef_construction);
    defer hnsw.deinit();

    // Insert some points
    try hnsw.insert(&[_]f32{ 1, 2, 3 });
    try hnsw.insert(&[_]f32{ 4, 5, 6 });
    try hnsw.insert(&[_]f32{ 7, 8, 9 });

    // Search for nearest neighbors
    const query = &[_]f32{ 3, 4, 5 };
    const results = try hnsw.search(query, 2, ef_search);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 2), results.len);
    try testing.expect(euclideanDistance(query, results[0].point) <= euclideanDistance(query, results[1].point));
}

test "HNSW - Empty Index" {
    const ef_construction = 200;
    const ef_search = 50;
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, ef_construction);
    defer hnsw.deinit();

    const query = &[_]f32{ 1, 2, 3 };
    const results = try hnsw.search(query, 5, ef_search);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 0), results.len);
}

test "HNSW - Single Point" {
    const ef_construction = 200;
    const ef_search = 50;
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, ef_construction);
    defer hnsw.deinit();

    const point = &[_]f32{ 1, 2, 3 };
    try hnsw.insert(point);

    const results = try hnsw.search(point, 1, ef_search);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqualSlices(f32, point, results[0].point);
}

test "HNSW - Exact Points" {
    const ef_construction = 200;
    const ef_search = 50;
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, ef_construction);
    defer hnsw.deinit();

    const point1 = &[_]f32{ 1, 2, 3 };
    const point2 = &[_]f32{ 4, 5, 6 };
    const point3 = &[_]f32{ 7, 8, 9 };
    try hnsw.insert(point1);
    try hnsw.insert(point2);
    try hnsw.insert(point3);

    const results = try hnsw.search(point1, 1, ef_search);
    defer allocator.free(results);
    try testing.expectEqual(@as(usize, 1), results.len);
    try testing.expectEqualSlices(f32, point1, results[0].point);

    const results2 = try hnsw.search(point2, 1, ef_search);
    defer allocator.free(results2);
    try testing.expectEqual(@as(usize, 1), results2.len);
    try testing.expectEqualSlices(f32, point2, results2[0].point);

    const results3 = try hnsw.search(point3, 1, ef_search);
    defer allocator.free(results3);
    try testing.expectEqual(@as(usize, 1), results3.len);
    try testing.expectEqualSlices(f32, point3, results3[0].point);
}

test "HNSW - Large Dataset" {
    const ef_construction = 200;
    const ef_search = 50;
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, ef_construction);
    defer hnsw.deinit();

    const num_points = 10000;
    const dim = 128;

    // Insert many points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    // Search for nearest neighbors
    const query = try randomPoint(allocator, dim);
    defer allocator.free(query);

    const k = 10;
    const results = try hnsw.search(query, k, ef_search);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, k), results.len);

    // Check if results are sorted by distance
    var last_dist: f32 = 0;
    for (results) |result| {
        const dist = euclideanDistance(query, result.point);
        try testing.expect(dist >= last_dist);
        last_dist = dist;
    }
}

test "HNSW - Edge Cases" {
    const ef_construction = 200;
    const ef_search = 100;
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, ef_construction);
    defer hnsw.deinit();

    // Insert duplicate points
    const point = &[_]f32{ 1, 2, 3 };
    try hnsw.insert(point);
    try hnsw.insert(point);

    const results = try hnsw.search(point, 2, ef_search);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 2), results.len);
    try testing.expectEqualSlices(f32, point, results[0].point);
    try testing.expectEqualSlices(f32, point, results[1].point);

    // Search with k larger than number of points
    const large_k_results = try hnsw.search(point, 100, ef_search);
    defer allocator.free(large_k_results);

    try testing.expectEqual(@as(usize, 2), large_k_results.len);
}

test "HNSW - Memory Leaks" {
    const ef_construction = 200;
    const ef_search = 100;
    var hnsw: HNSW(f32) = undefined;
    {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();

        hnsw = HNSW(f32).init(allocator, 16, ef_construction);

        const num_points = 1000;
        const dim = 64;

        for (0..num_points) |_| {
            const point = try randomPoint(allocator, dim);
            try hnsw.insert(point);
            // Intentionally not freeing 'point' to test if HNSW properly manages memory
        }

        const query = try randomPoint(allocator, dim);
        const results = try hnsw.search(query, 10, ef_search);
        _ = results;
        // Intentionally not freeing 'results' or 'query'
    }
    // The ArenaAllocator will detect any memory leaks when it's deinitialized
}

test "HNSW - Concurrent Access" {
    const ef_construction = 200;
    const ef_search = 100;
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, ef_construction);
    defer hnsw.deinit();

    const num_threads = 8;
    const points_per_thread = 1000;
    const dim = 128;

    const ThreadContext = struct {
        hnsw: *HNSW(f32),
        allocator: std.mem.Allocator,
    };

    const thread_fn = struct {
        fn func(ctx: *const ThreadContext) !void {
            for (0..points_per_thread) |_| {
                const point = try ctx.allocator.alloc(f32, dim);
                defer ctx.allocator.free(point);
                for (point) |*v| {
                    v.* = std.crypto.random.float(f32);
                }
                try ctx.hnsw.insert(point);
            }
        }
    }.func;

    var threads: [num_threads]std.Thread = undefined;
    var contexts: [num_threads]ThreadContext = undefined;

    for (&threads, 0..) |*thread, i| {
        contexts[i] = .{
            .hnsw = &hnsw,
            .allocator = allocator,
        };
        thread.* = try std.Thread.spawn(.{}, thread_fn, .{&contexts[i]});
    }

    for (&threads) |*thread| {
        thread.join();
    }

    // Verify that all points were inserted
    const expected_count = num_threads * points_per_thread;
    const actual_count = hnsw.nodes.count();
    try testing.expectEqual(expected_count, actual_count);

    // Test search after concurrent insertion
    const query = try randomPoint(allocator, dim);
    defer allocator.free(query);

    const results = try hnsw.search(query, 10, ef_search);
    defer allocator.free(results);

    try testing.expectEqual(@as(usize, 10), results.len);
}

test "HNSW - Stress Test" {
    const ef_construction = 200;
    const ef_search = 100;
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, ef_construction);
    defer hnsw.deinit();

    const num_points = 100000;
    const dim = 128;
    const num_queries = 100;

    // Insert many points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    // Perform multiple searches
    for (0..num_queries) |_| {
        const query = try randomPoint(allocator, dim);
        defer allocator.free(query);

        const results = try hnsw.search(query, 10, ef_search);
        defer allocator.free(results);

        try testing.expectEqual(@as(usize, 10), results.len);
    }
}

test "HNSW - Different Data Types" {
    const allocator = testing.allocator;

    // Test with integer type
    {
        const ef_construction = 200;
        const ef_search = 50;
        var hnsw_int = HNSW(i32).init(allocator, 16, ef_construction);
        defer hnsw_int.deinit();

        try hnsw_int.insert(&[_]i32{ 1, 2, 3 });
        try hnsw_int.insert(&[_]i32{ 4, 5, 6 });
        try hnsw_int.insert(&[_]i32{ 7, 8, 9 });

        const query_int = &[_]i32{ 3, 4, 5 };
        const results_int = try hnsw_int.search(query_int, 2, ef_search);
        defer allocator.free(results_int);

        try testing.expectEqual(@as(usize, 2), results_int.len);
    }

    // Test with float64 type
    {
        const ef_construction = 200;
        const ef_search = 50;
        var hnsw_f64 = HNSW(f64).init(allocator, 16, ef_construction);
        defer hnsw_f64.deinit();

        try hnsw_f64.insert(&[_]f64{ 1.1, 2.2, 3.3 });
        try hnsw_f64.insert(&[_]f64{ 4.4, 5.5, 6.6 });
        try hnsw_f64.insert(&[_]f64{ 7.7, 8.8, 9.9 });

        const query_f64 = &[_]f64{ 3.3, 4.4, 5.5 };
        const results_f64 = try hnsw_f64.search(query_f64, 2, ef_search);
        defer allocator.free(results_f64);

        try testing.expectEqual(@as(usize, 2), results_f64.len);
    }
}

test "HNSW - Consistency" {
    const ef_construction = 200;
    const ef_search = 100;
    const allocator = testing.allocator;
    var hnsw = HNSW(f32).init(allocator, 16, ef_construction);
    defer hnsw.deinit();

    const num_points = 10000;
    const dim = 128;

    // Insert points
    for (0..num_points) |_| {
        const point = try randomPoint(allocator, dim);
        defer allocator.free(point);
        try hnsw.insert(point);
    }

    // Perform multiple searches with the same query
    const query = try randomPoint(allocator, dim);
    defer allocator.free(query);

    const num_searches = 10;
    const k = 10;
    var first_result = try allocator.alloc(f32, k * dim);
    defer allocator.free(first_result);

    for (0..num_searches) |i| {
        const results = try hnsw.search(query, k, ef_search);
        defer allocator.free(results);

        if (i == 0) {
            // Store the first result for comparison
            for (results, 0..) |result, j| {
                @memcpy(first_result[j * dim .. (j + 1) * dim], result.point);
            }
        } else {
            // Compare with the first result
            for (results, 0..) |result, j| {
                const start = j * dim;
                const end = (j + 1) * dim;
                try testing.expectEqualSlices(f32, first_result[start..end], result.point);
            }
        }
    }
}
