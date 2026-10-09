import * as np from "https://esm.sh/numpy-ts@1.7.0"
import { arr } from "https://gnlow.dev/util@0.1.2"
import { FunPlane } from "https://gnlow.dev/plane@0.1.5"

type Vec<N extends number> = number[] & { length: N }

export const daviesHarte =
<N extends number>
(dimension: Vec<N>, kernel: (d: number) => number, seed = 42) => {
    np.random.seed(seed)

    const covData = np.fromfunction((...cs) => {
        const ds = cs.map((c, i) =>
            c <= dimension[i]/2 ? c : dimension[i]-c
        )
        return kernel(Math.hypot(...ds))
    }, dimension)
    const covGrid = np.array(covData)

    const covFft = np.fft.fft2(covGrid)
    const realCov = np.real(covFft)
    const safeCov = np.clip(realCov, 0, null)
    const sqrtCov = np.sqrt(safeCov)
    
    const noise = np.random.normal(0, 1, dimension) as np.NDArray<"float32">
    const noiseFft = np.fft.fft2(noise)

    const filteredFft = np.multiply(noiseFft, sqrtCov)
    
    const rawField = np.real(np.fft.ifft2(filteredFft))
    return new Field<N>(rawField)
}

export class Field<N extends number> {
    constructor(public raw: np.NDArray<"float32">) {}
    get w() { return this.raw.shape[0] }
    get h() { return this.raw.shape[1] }
    slice(x: string, y: string) {
        return new Field(this.raw.slice(x, y))
    }
    get(coord: Vec<N>) {
        return this.raw.get(coord)
    }
    toPlane(this: Field<2>) {
        return new FunPlane(this.w, this.h, ([x, y]) => this.get([x, y]))
    }
    project<M extends number>
    (target: Vec<M>, f: (...cs: Vec<M>) => Vec<N>) {
        return new Field(np.fromfunction((...cs) =>
            this.get(f(cs))
        ), target)
    }
    equirect(
        this: Field<3>,
        o: {
            center: Vec<3>,
            radius: number,
            res: Vec<2>,
        },
    ) {
        const [w, h] = o.res
        return this.project([w, h], ([u, v]) => {
            const theta = (u / w) * 2 * Math.PI - Math.PI
            const phi = Math.PI / 2 - (v / h) * Math.PI

            const x = center[0] + radius * Math.cos(phi) * Math.cos(theta)
            const y = center[1] + radius * Math.cos(phi) * Math.sin(theta)
            const z = center[2] + radius * Math.sin(phi)

            return [x, y, z]
        })
    }
}

export const kernels = {
    gaussian: (r: number) => (d: number) => Math.exp(-(d**2)/r**2),
    triangular: (r: number) => (d: number) => Math.max(0, 1-d/r),
    exponential: (r: number) => (d: number) => Math.exp(-d/r),
}
