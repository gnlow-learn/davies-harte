import * as np from "https://esm.sh/numpy-ts@1.7.0"
import { arr } from "https://gnlow.dev/util@0.1.2"
import { FunPlane } from "https://gnlow.dev/plane@0.1.5"

export const daviesHarte =
(w: number, h: number, kernel: (d: number) => number, seed = 42) => {
    np.random.seed(seed)

    w /= 2
    h /= 2
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
    return new Field(rawField)
}

export class Field {
    constructor(public raw: np.NDArray<"float32">) {}
    get w() { return this.raw.shape[0] }
    get h() { return this.raw.shape[1] }
    slice(x: string, y: string) {
        return new Field(this.raw.slice(x, y))
    }
    get(coord: number[]) {
        return this.raw.get(coord)
    }
    toPlane() {
        return new FunPlane(this.w, this.h, ([x, y]) => this.get([x, y]))
    }
}

export const kernels = {
    gaussian: (r: number) => (d: number) => Math.exp(-(d**2)/r**2),
    triangular: (r: number) => (d: number) => Math.max(0, 1-d/r),
    exponential: (r: number) => (d: number) => Math.exp(-d/r),
}
