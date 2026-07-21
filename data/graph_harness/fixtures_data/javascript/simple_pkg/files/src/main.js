import React from "react";                 // bare/external -> node_modules, NO edge
import { helper } from "./util/helper";    // -> src/util/helper
import * as models from "./models";        // directory import -> src/models/index

const { VALUE } = require("./consts");      // CommonJS require intra-repo -> src/consts

export { CONFIG } from "./consts";          // re-export -> src/consts (dedup w/ require)

// A backtick template containing an import STRING must NOT be detected:
const codegen = `import fake from "./ghost";`;

export async function main() {
  const dyn = await import("./util/helper");  // dynamic import -> src/util/helper (dup)
  const m = new models.Model();
  return helper(m, React, VALUE) + dyn.helper.length + codegen.length;
}
