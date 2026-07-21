import { VALUE } from "../consts";        // relative up-dir ESM -> src/consts
const lodash = require("lodash");         // bare CommonJS require -> external, NO edge
// import { nope } from "./ghost";        // in a comment -> NOT an edge

export function helper(model, react, base) {
  return base + VALUE + Object.keys(lodash).length;
}
