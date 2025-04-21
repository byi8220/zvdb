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
            const level = self.randomLevel();
            var node = try Node.init(self.allocator, id, point, level);
            errdefer node.deinit(self.allocator);

            try self.nodes.put(id, node);

            if (self.entry_point) |entry| {
                var ep_copy = entry;
                var curr_dist = distance(node.point, self.nodes.get(ep_copy).?.point);

                for (0..self.max_level + 1) |layer| {
                    var changed = true;
                    while (changed) {
                        changed = false;
                        const curr_node = self.nodes.get(ep_copy).?;
                        if (layer < curr_node.connections.len) {
                            for (curr_node.connections[layer].items) |neighbor_id| {
                                const neighbor = self.nodes.get(neighbor_id).?;
                                const dist = distance(node.point, neighbor.point);
                                if (dist < curr_dist) {
                                    ep_copy = neighbor_id;
                                    curr_dist = dist;
                                    changed = true;
                                }
                            }
                        }
                    }

                    if (layer <= level) {
                        try self.connect(id, ep_copy, @intCast(layer));
                    }
                }
            } else {
                self.entry_point = id;
            }

            if (level > self.max_level) {
                self.max_level = level;
            }
        }

        fn connect(self: *Self, source: usize, target: usize, level: usize) !void {
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
                try self.shrinkConnections(source, level);
            }
            if (level < target_node.connections.len) {
                try self.shrinkConnections(target, level);
            }
        }

        fn shrinkConnections(self: *Self, node_id: usize, level: usize) !void {
            var node = self.nodes.getPtr(node_id).?;
            var connections = &node.connections[level];
            if (connections.items.len <= self.m) return;

            var candidates = try self.allocator.alloc(usize, connections.items.len);
            defer self.allocator.free(candidates);
            @memcpy(candidates, connections.items);

            const Context = struct {
                self: *Self,
                node: *Node,
            };
            const context = Context{ .self = self, .node = node };

            const compareFn = struct {
                fn compare(ctx: Context, a: usize, b: usize) bool {
                    const dist_a = distance(ctx.node.point, ctx.self.nodes.get(a).?.point);
                    const dist_b = distance(ctx.node.point, ctx.self.nodes.get(b).?.point);
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
            const max_level = 31;
            while (level < max_level and std.crypto.random.float(f32) < 0.5) {
                level += 1;
            }
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

        pub fn search(self: *Self, query: []const T, k: usize) ![]const Node {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.entry_point == null) {
                var result = try ArrayList(Node).initCapacity(self.allocator, k);
                errdefer result.deinit();
                return result.toOwnedSlice();
            }

            const entry_points = [1]usize{self.entry_point.?};
            // TODO: Implement HNSW search. This is currently just brute force k nearest neighbors on layer 0
            return self.searchLayer(query, &entry_points, k, 0);
        }

        // Implementation of Search-layer algorithm from https://arxiv.org/pdf/1603.09320
        fn searchLayer(self: *Self, query: []const T, entry_points: []const usize, k: usize, level: usize) ![]const Node {
            var result = try ArrayList(Node).initCapacity(self.allocator, k);
            errdefer result.deinit();

            if (entry_points.len == 0) return result.toOwnedSlice();

            var candidates = std.PriorityQueue(CandidateNode, void, CandidateNode.lessThan).init(self.allocator, {});
            defer candidates.deinit();

            var w = std.PriorityQueue(CandidateNode, void, CandidateNode.greaterThan).init(self.allocator, {});
            defer w.deinit();

            var visited = std.AutoHashMap(usize, void).init(self.allocator);
            defer visited.deinit();

            for (entry_points) |entry| {
                const dist = distance(query, self.nodes.get(entry).?.point);
                try candidates.add(.{ .id = entry, .distance = dist });
                try visited.put(entry, {});
                try w.add(.{ .id = entry, .distance = dist });
            }

            while (candidates.count() > 0) {
                // current = closest candidate
                const current = candidates.remove();
                const current_node = self.nodes.get(current.id).?;
                const current_dist = distance(query, current_node.point);

                const f = w.peek().?; // Always non-empty; We never pop unless we have more than k results
                if (current_dist > f.distance) {
                    break;
                }

                for (current_node.connections[level].items) |neighbor_id| {
                    if (visited.contains(neighbor_id)) continue;
                    try visited.put(neighbor_id, {});
                    const neighbor = self.nodes.get(neighbor_id).?;
                    const dist = distance(query, neighbor.point);
                    if (w.count() < k or dist < f.distance) {
                        try candidates.add(.{ .id = neighbor_id, .distance = dist });
                        try w.add(.{ .id = neighbor_id, .distance = dist });
                        if (w.count() > k) {
                            _ = w.remove();
                        }
                    }
                }
            }

            const Context = struct {
                query: []const T,
                pub fn lessThan(ctx: @This(), a: Node, b: Node) bool {
                    return distance(ctx.query, a.point) < distance(ctx.query, b.point);
                }
            };

            // Transfer `w` to `result`.
            // Side note: Kinda wish PriorityQueue had a `toOwnedSlice()` method, so I can just move the heap over then sort.
            for (w.items) |item| {
                const node = self.nodes.get(item.id).?;
                try result.append(node);
            }
            std.sort.insertion(Node, result.items, Context{ .query = query }, Context.lessThan);

            return result.toOwnedSlice();
        }

        const CandidateNode = struct {
            id: usize,
            distance: T,

            fn lessThan(_: void, a: CandidateNode, b: CandidateNode) std.math.Order {
                return std.math.order(a.distance, b.distance);
            }
            fn greaterThan(_: void, a: CandidateNode, b: CandidateNode) std.math.Order {
                return std.math.order(b.distance, a.distance);
            }
        };
    };
}
