import { daviesHarte, kernels } from "./mod.ts"
import { FunPlane } from "https://gnlow.dev/plane@0.1.5"

const gen =
(kernel: keyof typeof kernels) => {
    const p32 = daviesHarte(256, 256, kernels[kernel](20))
    
    const plane = new FunPlane(256, 256, ([x, y]) => p32.get([x, y]))
    const png = plane.map(x => 128*(1+Math.tanh(x as number))).grayscale().toPng()
    return Deno.writeFile(`${kernel}.png`, png)
}

await gen("triangular")
await gen("gaussian")
await gen("exponential")
