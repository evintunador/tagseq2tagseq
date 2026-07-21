const std = @import("std");
const helper = @import("util/helper.zig");
const bar = @import("sub/bar.zig");
const consts = @import("consts.zig");

pub fn main() void {
    _ = std;
    _ = helper;
    _ = bar;
    _ = consts;
}
