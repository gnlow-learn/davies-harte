import { daviesHarte, kernels } from "./mod.ts"
import { FunPlane } from "https://gnlow.dev/plane@0.1.5"
import { Cmap } from "https://gnlow.dev/cmap@0.0.0-beta.4"

const terrain = Cmap.fromStops([
    { pos: 0.00, rgb: [ 10,  30, 100] },
    { pos: 0.60, rgb: [ 40, 100, 180] },
    { pos: 0.65, rgb: [100, 180, 220] }, // light blue
    //{ pos: 0.68, rgb: [210, 210, 150] }, // sand
    { pos: 0.70, rgb: [100, 160, 100] }, // light green
    { pos: 0.80, rgb: [ 60, 140,  50] }, // dark green
    { pos: 0.88, rgb: [180, 140,  70] }, // light brown
    { pos: 0.90, rgb: [120,  70,  30] }, // dark brown
    { pos: 1.00, rgb: [255, 255, 255] }, // white
]).clamp()


const gen =
(kernel: keyof typeof kernels) => {
    const p32 = daviesHarte(256, 256, kernels[kernel](80))
    
    const plane = new FunPlane(256, 256, ([x, y]) => p32.get([x, y]))
    const png = plane.map(p => [
        ...terrain
            .at((1+Math.tanh(p as number / 2))/2),
        255,
    ]).upscale(2).toPng()
    return Deno.writeFile(`${kernel}.png`, png)
}

await gen("triangular")
await gen("gaussian")
await gen("exponential")
