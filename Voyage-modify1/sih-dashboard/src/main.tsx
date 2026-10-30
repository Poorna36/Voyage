import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// WebGL2 specification conformance patch:
// UNPACK_FLIP_Y_WEBGL and UNPACK_PREMULTIPLY_ALPHA_WEBGL are not allowed
// for 3D textures (TEXTURE_3D or TEXTURE_2D_ARRAY).
if (typeof window !== 'undefined' && window.WebGL2RenderingContext) {
  const origTexImage3D = WebGL2RenderingContext.prototype.texImage3D;
  WebGL2RenderingContext.prototype.texImage3D = function (
    this: WebGL2RenderingContext,
    ...args: any[]
  ) {
    const flipY = this.getParameter(this.UNPACK_FLIP_Y_WEBGL);
    const premult = this.getParameter(this.UNPACK_PREMULTIPLY_ALPHA_WEBGL);
    if (flipY) this.pixelStorei(this.UNPACK_FLIP_Y_WEBGL, false);
    if (premult) this.pixelStorei(this.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    try {
      return (origTexImage3D as any).apply(this, args);
    } finally {
      if (flipY) this.pixelStorei(this.UNPACK_FLIP_Y_WEBGL, flipY);
      if (premult) this.pixelStorei(this.UNPACK_PREMULTIPLY_ALPHA_WEBGL, premult);
    }
  };
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
