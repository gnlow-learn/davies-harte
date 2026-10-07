import * as np from "https://esm.sh/numpy-ts@1.7.0"
import { arr } from "https://gnlow.dev/util@0.1.2"

export const daviesHarte =
(w: number, h: number, r = 15) => {
    const covData = arr(2*w).map(i => arr(2*h).map(j => {
        const dx = i <= w ? i : 2*w-i
        const dy = j <= h ? j : 2*h-j
        const dist = Math.hypot(dx, dy)
        return Math.max(0, 1-dist/r)
    }))
    const covGrid = np.array(covData)

    const covFft = np.fft.fft2(covGrid)
    const realCov = np.real(covFft)
    const safeCov = np.clip(realCov, 0, null)
    const sqrtCov = np.sqrt(safeCov)
    
    const noise = np.random.normal(0, 1, [2*w, 2*h])
    const noiseFft = np.fft.fft2(noise)

    const filteredFft = np.multiply(noiseFft, sqrtCov)
    
    const rawField = np.real(np.fft.ifft2(filteredFft))
    return rawField.slice(`0:${w}`, `0:${h}`)
}

console.log(daviesHarte(32, 32))
