const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const AutoHashMap = std.AutoHashMap;
const Order = std.math.Order;
const Mutex = std.Thread.Mutex;

pub fn HNSW(comptime T: type) type {
    return struct {
        const Self = @This();

        const Node = struct {
            id: usize,
            point: []T,
            connections: []ArrayList(usize),
            mutex: Mutex,

            fn init(allocator: Allocator, id: usize, point: []const T, level: usize) !Node {
                const connections = try allocator.alloc(ArrayList(usize), level + 1);
                errdefer allocator.free(connections);
                for (connections) |*conn| {
                    conn.* = ArrayList(usize).init(allocator);
                }
                const owned_point = try allocator.alloc(T, point.len);
                errdefer allocator.free(owned_point);
                @memcpy(owned_point, point);
                return Node{
                    .id = id,
                    .point = owned_point,
                    .connections = connections,
                    .mutex = Mutex{},
                };
            }

            fn deinit(self: *Node, allocator: Allocator) void {
                for (self.connections) |*conn| {
                    conn.deinit();
                }
                allocator.free(self.connections);
                allocator.free(self.point);
            }
        };

        allocator: Allocator,
        nodes: AutoHashMap(usize, Node),
        entry_point: ?usize,
        max_level: usize,
        m: usize,
        ef_construction: usize,
        mutex: Mutex,

        pub fn init(allocator: Allocator, m: usize, ef_construction: usize) Self {
            return .{
                .allocator = allocator,
                .nodes = AutoHashMap(usize, Node).init(allocator),
                .entry_point = null,
                .max_level = 0,
                .m = m,
                .ef_construction = ef_construction,
                .mutex = Mutex{},
            };
        }

        pub fn deinit(self: *Self) void {
            var it = self.nodes.iterator();
            while (it.next()) |entry| {
                var node = entry.value_ptr;
                node.deinit(self.allocator);
            }
            self.nodes.deinit();
        }

        pub fn insert(self: *Self, point: []const T) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            const id = self.nodes.count();
            const level_to_place = self.randomLevel();

            // Initialization case when graph is empty.
            // We don't strictly have to handle this case (we could just make it fall through)
            if (self.entry_point == null) {
                self.entry_point = id;
                self.max_level = level_to_place;
                var node = try Node.init(self.allocator, id, point, level_to_place);
                errdefer node.deinit(self.allocator);
                try self.nodes.put(id, node);
                return;
            }

            // Normal case.
            var node = try Node.init(self.allocator, id, point, level_to_place);
            errdefer node.deinit(self.allocator);
            try self.nodes.put(id, node);

            var level = self.max_level;
            var w: []const Node = undefined;
            var entry_points = [_]usize{self.entry_point.?};
            while (level > level_to_place) : (level -= 1) {
                w = try self.searchLayer(point, &entry_points, 1, level);
                entry_points[0] = w[0].id;
                defer self.allocator.free(w);
            }
            const top_insertion_level = @min(self.max_level, level_to_place);
            for (0..top_insertion_level + 1) |i| {
                level = top_insertion_level - i;
                w = try self.searchLayer(point, &entry_points, self.ef_construction, level);
                defer self.allocator.free(w);
                entry_points[0] = w[0].id;
                // Attach the closest `m` neighbors
                // TODO: Add heuristic selection
                const m = if (level == 0) self.m * 2 else self.m;
                const num_neighbors = @min(m, w.len);
                for (w[0..num_neighbors]) |neighbor| {
                    try self.connect(id, neighbor.id, level, m);
                }
            }

            if (level_to_place > self.max_level) {
                self.max_level = level_to_place;
                self.entry_point = id;
            }
        }

        fn connect(self: *Self, source: usize, target: usize, level: usize, m: usize) !void {
            var source_node = self.nodes.getPtr(source) orelse return error.NodeNotFound;
            var target_node = self.nodes.getPtr(target) orelse return error.NodeNotFound;

            source_node.mutex.lock();
            defer source_node.mutex.unlock();
            target_node.mutex.lock();
            defer target_node.mutex.unlock();

            if (level < source_node.connections.len) {
                try source_node.connections[level].append(target);
            }
            if (level < target_node.connections.len) {
                try target_node.connections[level].append(source);
            }

            if (level < source_node.connections.len) {
                try self.shrinkConnections(source, level, m);
            }
            if (level < target_node.connections.len) {
                try self.shrinkConnections(target, level, m);
            }
        }

        fn shrinkConnections(self: *Self, node_id: usize, level: usize, m: usize) !void {
            var node = self.nodes.getPtr(node_id).?;
            var connections = &node.connections[level];
            if (connections.items.len <= m) return;

            var candidates = try self.allocator.alloc(usize, connections.items.len);
            defer self.allocator.free(candidates);
            @memcpy(candidates, connections.items);

            const Context = struct {
                self: *Self,
                node: *Node,
            };
            const context = Context{ .self = self, .node = node };

            // TODO: Can this be optimized away? Have we computed these all before?
            const compareFn = struct {
                fn compare(ctx: Context, a: usize, b: usize) bool {
                    const dist_a = distance_simd(ctx.node.point, ctx.self.nodes.get(a).?.point);
                    const dist_b = distance_simd(ctx.node.point, ctx.self.nodes.get(b).?.point);
                    return dist_a < dist_b;
                }
            }.compare;

            std.sort.insertion(usize, candidates, context, compareFn);

            connections.shrinkRetainingCapacity(self.m);
            @memcpy(connections.items, candidates[0..self.m]);
        }

        fn randomLevel(self: *Self) usize {
            _ = self;
            var level: usize = 0;
            // Paper used log_e, but maybe using log_2 is ok & better for perf?
            const max_level = 40;
            const ml: f32 = 1.0 / std.math.log2(@as(f32, max_level));
            const rand_level = -std.math.log2(std.crypto.random.float(f32)) * ml;
            level = @intFromFloat(rand_level);
            return level;
        }

        fn distance(a: []const T, b: []const T) T {
            if (a.len != b.len) {
                @panic("Mismatched dimensions in distance calculation");
            }
            var sum: T = 0;
            for (a, 0..) |_, i| {
                const diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum; // Note: We're returning squared distance for efficiency
        }

        fn distance_simd(a: []const T, b: []const T) T {
            // This is likely the hottest function in the code. Optimizing (or avoiding) this probably nets the biggest wins.
            if (a.len != b.len) {
                @panic("Mismatched dimensions in distance calculation");
            }
            var sum: T = 0;
            const vec_width = 8; // Hardcoded 8x16 for AVX2
            const n_vecs = a.len / vec_width;
            // First handle full vector chunks
            for (0..n_vecs) |v| {
                const i = v * vec_width;
                var vec_a: @Vector(vec_width, T) = undefined;
                var vec_b: @Vector(vec_width, T) = undefined;
                for (0..vec_width) |j| {
                    vec_a[j] = a[i + j];
                    vec_b[j] = b[i + j];
                }
                vec_a = vec_a - vec_b;
                sum += @reduce(.Add, vec_a * vec_a);
            }
            // Handle remaining elements
            for (n_vecs * vec_width..a.len) |i| {
                const diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum; // Note: We're returning squared distance for efficiency
        }

        // Implementation of K-NN search algorithm
        pub fn search(self: *Self, query: []const T, k: usize, ef_search: usize) ![]const Node {
            if (k > ef_search) {
                @panic("ef_search must be greater than or equal to k");
            }
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.entry_point == null) {
                var result = try ArrayList(Node).initCapacity(self.allocator, k);
                errdefer result.deinit();
                return result.toOwnedSlice();
            }
            var w: []const Node = undefined;
            var entry_points = [_]usize{self.entry_point.?};
            var level = self.max_level;
            while (level > 0) : (level -= 1) {
                w = try self.searchLayer(query, &entry_points, 1, level);
                defer self.allocator.free(w);
                entry_points[0] = w[0].id;
            }
            w = try self.searchLayer(query, &entry_points, ef_search, 0);
            defer self.allocator.free(w);
            const numel = @min(k, w.len);
            const result = try self.allocator.alloc(Node, numel);
            errdefer self.allocator.free(result);
            @memcpy(result, w[0..numel]);
            return result;
        }

        // Implementation of Search-layer algorithm from https://arxiv.org/pdf/1603.09320
        fn searchLayer(self: *Self, query: []const T, entry_points: []const usize, ef: usize, level: usize) ![]const Node {
            var result = try ArrayList(Node).initCapacity(self.allocator, ef);
            errdefer result.deinit();

            if (entry_points.len == 0) return result.toOwnedSlice();

            var candidates = std.PriorityQueue(CandidateNode, void, CandidateNode.lessThanOrdering).init(self.allocator, {});
            defer candidates.deinit();

            var w = std.PriorityQueue(CandidateNode, void, CandidateNode.greaterThanOrdering).init(self.allocator, {});
            defer w.deinit();

            var visited = std.AutoHashMap(usize, void).init(self.allocator);
            defer visited.deinit();

            for (entry_points) |entry| {
                const dist = distance_simd(query, self.nodes.get(entry).?.point);
                try candidates.add(.{ .id = entry, .distance = dist });
                try visited.put(entry, {});
                try w.add(.{ .id = entry, .distance = dist });
            }

            while (candidates.count() > 0) {
                // current = closest candidate
                const current = candidates.remove();
                const current_node = self.nodes.get(current.id).?;
                const current_dist = current.distance;

                const f = w.peek().?; // Always non-empty; We never pop unless we have more than k results
                if (current_dist > f.distance) {
                    break;
                }

                for (current_node.connections[level].items) |neighbor_id| {
                    if (visited.contains(neighbor_id)) continue;
                    try visited.put(neighbor_id, {});
                    const neighbor = self.nodes.get(neighbor_id).?;
                    const dist = distance_simd(query, neighbor.point);
                    if (w.count() < ef or dist < f.distance) {
                        try candidates.add(.{ .id = neighbor_id, .distance = dist });
                        try w.add(.{ .id = neighbor_id, .distance = dist });
                        if (w.count() > ef) {
                            _ = w.remove();
                        }
                    }
                }
            }

            std.sort.heap(CandidateNode, w.items, {}, CandidateNode.lessThan);
            for (w.items) |item| {
                const node = self.nodes.get(item.id).?;
                try result.append(node);
            }

            return result.toOwnedSlice();
        }

        const CandidateNode = struct {
            id: usize,
            distance: T,

            fn lessThan(_: void, a: CandidateNode, b: CandidateNode) bool {
                return a.distance < b.distance;
            }
            fn lessThanOrdering(_: void, a: CandidateNode, b: CandidateNode) std.math.Order {
                return std.math.order(a.distance, b.distance);
            }
            fn greaterThanOrdering(_: void, a: CandidateNode, b: CandidateNode) std.math.Order {
                return std.math.order(b.distance, a.distance);
            }
        };
    };
}
