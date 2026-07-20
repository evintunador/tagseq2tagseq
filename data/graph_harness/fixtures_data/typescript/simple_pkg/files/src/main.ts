import React from "react";               // bare/external -> node_modules, NO edge
import { helper } from "./util/helper";  // -> src/util/helper
import * as models from "./models";      // directory import -> src/models/index
import type { Cfg } from "./consts";     // type-only import still creates a file edge -> src/consts

export { VALUE } from "./consts";        // re-export -> src/consts

export function main(): void {
  const m = new models.Model();
  helper(m, React, {} as Cfg);
}
