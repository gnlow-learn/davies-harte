import { daviesHarte, kernels } from "./mod.ts"
import { FunPlane } from "https://gnlow.dev/plane@0.1.5"

const p32 = daviesHarte(256, 256, kernels.gaussian(20))

const plane = new FunPlane(256, 256, ([x, y]) => p32.get([x, y]))
const png = plane.map(x => 128*(1+Math.tanh(x as number))).grayscale().toPng()
await Deno.writeFile("test.png", png)
