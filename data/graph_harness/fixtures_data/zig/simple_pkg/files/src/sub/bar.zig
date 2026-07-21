const std = @import("std");
const helper = @import("../util/helper.zig");

// A string literal that merely LOOKS like an import must not be detected:
const fake = "const x = @import(\"strlit.zig\");";

pub fn bar() u32 {
    _ = fake;
    return helper.help();
}
