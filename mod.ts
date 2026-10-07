import * as np from "https://esm.sh/numpy-ts@1.7.0"
import { arr } from "https://gnlow.dev/util@0.1.2"
import { FunPlane } from "https://gnlow.dev/plane@0.1.5"

export const daviesHarte =
(w: number, h: number, kernel: (d: number) => number) => {
    const covData = arr(2*w).map(i => arr(2*h).map(j => {
        const dx = i <= w ? i : 2*w-i
        const dy = j <= h ? j : 2*h-j
        const dist = Math.hypot(dx, dy)
        return kernel(dist)
    }))
    const covGrid = np.array(covData)

    const covFft = np.fft.fft2(covGrid)
    const realCov = np.real(covFft)
    const safeCov = np.clip(realCov, 0, null)
    const sqrtCov = np.sqrt(safeCov)
    
    const noise = np.random.normal(0, 1, [2*w, 2*h]) as np.NDArray<"float32">
    const noiseFft = np.fft.fft2(noise)

    const filteredFft = np.multiply(noiseFft, sqrtCov)
    
    const rawField = np.real(np.fft.ifft2(filteredFft))
    return rawField.slice(`0:${w}`, `0:${h}`)
}

const kernels = {
    gaussian: (r: number) => (d: number) => Math.exp(-(d**2)/r**2),
    triangular: (r: number) => (d: number) => Math.max(0, 1-d/r),
    exponential: (r: number) => (d: number) => Math.exp(-d/r),
}

const p32 = daviesHarte(256, 256, kernels.gaussian(20))

const plane = new FunPlane(256, 256, ([x, y]) => p32.get([x, y]))
const png = plane.map(x => 128*(1+Math.tanh(x as number))).grayscale().toPng()
await Deno.writeFile("test.png", png)
