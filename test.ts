import { daviesHarte, kernels } from "./mod.ts"
import { FunPlane } from "https://gnlow.dev/plane@0.1.5"
import { globe } from "https://gnlow.dev/cmap@0.0.0-beta.4"

const gen =
(kernel: keyof typeof kernels) => {
    const p32 = daviesHarte(256, 256, kernels[kernel](80))
    
    const plane = new FunPlane(256, 256, ([x, y]) => p32.get([x, y]))
    const png = plane.map(p => [...globe.range(-0.5, 1).at((1+Math.tanh(p as number / 2))/2), 255]).toPng()
    return Deno.writeFile(`${kernel}.png`, png)
}

await gen("triangular")
await gen("gaussian")
await gen("exponential")
