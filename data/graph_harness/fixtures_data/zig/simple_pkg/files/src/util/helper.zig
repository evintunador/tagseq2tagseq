const std = @import("std");
// Importing consts from one dir up. The commented-out @import("nope.zig") below
// must NOT be detected (it lives in a // comment):
// const nope = @import("nope.zig");
const consts = @import("../consts.zig");
const builtin = @import("builtin");

pub fn help() u32 {
    return consts.VALUE;
}
