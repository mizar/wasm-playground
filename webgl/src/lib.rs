use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{WebGlProgram, WebGl2RenderingContext, WebGlShader};

fn window() -> web_sys::Window {
    web_sys::window().expect("no global `window` exists")
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    window()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}

fn document() -> web_sys::Document {
    window()
        .document()
        .expect("should have a document on window")
}

fn body() -> web_sys::HtmlElement {
    document().body().expect("document should have a body")
}

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
  body().style().set_property("margin", "0")?;
  body().style().set_property("padding", "0")?;

  /*
  let p = document()
    .create_element("p")?
    .dyn_into::<web_sys::HtmlElement>()?;
  body().append_child(&p)?;
  */

  let canvas = document()
    .create_element("canvas")?
    .dyn_into::<web_sys::HtmlCanvasElement>()?;
  body().append_child(&canvas)?;

  let canvas_w: u32 = 1024;
  let canvas_h: u32 = 640;

  canvas.set_width(canvas_w);
  canvas.set_height(canvas_h);
  canvas.style().set_property("border", "solid")?;

  let context = canvas
    .get_context("webgl2")?
    .unwrap()
    .dyn_into::<WebGl2RenderingContext>()?;

  let vert_shader = compile_shader(
    &context,
    WebGl2RenderingContext::VERTEX_SHADER,
    r#"#version 300 es
    precision highp float;
    in vec4 position;
    out vec4 vPosition;
    void main() {
      vPosition = position;
      gl_Position = position;
    }
    "#,
  )?;
  let frag_shader = compile_shader(
    &context,
    WebGl2RenderingContext::FRAGMENT_SHADER,
    r#"#version 300 es
    precision highp float;
    uniform float time;
    uniform vec2 mouse;
    uniform vec2 resolution;
    in vec4 vPosition;
    out vec4 outColor;

    // sholder angle
    const float angle_a = radians(18.);
    // side angle
    const float angle_b = radians(9.);
    // surface angle
    const float angle_c = radians(4.5);

    const float sin_a = sin(angle_a);
    const float cos_a = cos(angle_a);
    const float tan_a = tan(angle_a);
    const float sin_b = sin(angle_b);
    const float cos_b = cos(angle_b);
    const float tan_b = tan(angle_b);
    const float sin_c = sin(angle_c);
    const float cos_c = cos(angle_c);
    const float cos_c_inv = 1.0 / cos(angle_c);
    const float tan_c = tan(angle_c);

    const vec3 vec3_zero = vec3(0., 0., 0.);
    const vec3 vec3_one = vec3(1., 1., 1.);

    float dot2( in vec3 v ) { return dot(v,v); }
    float udQuad( vec3 p, vec3 a, vec3 b, vec3 c, vec3 d )
    {
        vec3 ba = b - a; vec3 pa = p - a;
        vec3 cb = c - b; vec3 pb = p - b;
        vec3 dc = d - c; vec3 pc = p - c;
        vec3 ad = a - d; vec3 pd = p - d;
        vec3 nor = cross( ba, ad );

        return sqrt(
        (sign(dot(cross(ba,nor),pa)) +
        sign(dot(cross(cb,nor),pb)) +
        sign(dot(cross(dc,nor),pc)) +
        sign(dot(cross(ad,nor),pd))<3.0)
        ?
        min( min( min(
        dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
        dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
        dot2(dc*clamp(dot(dc,pc)/dot2(dc),0.0,1.0)-pc) ),
        dot2(ad*clamp(dot(ad,pd)/dot2(ad),0.0,1.0)-pd) )
        :
        dot(nor,pa)*dot(nor,pa)/dot2(nor) );
    }

    float sdPiece_simple(vec3 p, vec3 g_i) {
      // boundbox
      vec3 pg = abs(p) - g_i;
      float bmax = max(pg.x, max(pg.y, pg.z));
      if (bmax >= 0.01) return bmax;

      const float xyr = 0.03;
      const float zr = 0.01;
      const vec3 r = vec3(vec2(xyr), zr);
      const vec3 xyzm = vec3(vec2(zr / xyr), 1.0);
      const vec3 xymz0 = vec3(vec2(zr / xyr), 0.0);
      vec3 g = g_i - r;
      float sholl = (g.x - g.y * 2.0 * tan_b) / (cos_a - sin_a * tan_b);
      float sholw = sholl * cos_a;
      float sholh = sholl * sin_a;
      float sidel = (g.y * 2.0 - g.x * tan_a) / (cos_b - sin_b * tan_a);
      float sidew = sidel * sin_b;
      float sideh = sidel * cos_b;
      float head_thick = g.z - g.y * 2.0 * tan_c;
      float shol_thick = g.z - sideh * tan_c;

      vec3 v3_pabs = vec3(abs(p.x), p.y, abs(p.z));
      vec3 v3_head = vec3(0.0, g.y, head_thick);
      vec3 v3_shol = vec3(sholw, (sideh - g.y), shol_thick);
      vec3 v3_bedg = vec3(g.x, -g.y, g.z);
      vec3 v3_bcen = vec3(0, -g.y, g.z);

      vec3 v3_pabsm = v3_pabs * xyzm;
      vec3 v3_headm = v3_head * xyzm;
      vec3 v3_sholm = v3_shol * xyzm;
      vec3 v3_bedgm = v3_bedg * xyzm;
      vec3 v3_bcenm = v3_bcen * xyzm;

      vec3 v3_head0 = v3_head * xymz0;
      vec3 v3_shol0 = v3_shol * xymz0;
      vec3 v3_bedg0 = v3_bedg * xymz0;
      vec3 v3_bcen0 = v3_bcen * xymz0;

      return min(min(min(
        udQuad(v3_pabsm, v3_headm, v3_bcenm, v3_bedgm, v3_sholm),
        udQuad(v3_pabsm, v3_headm, v3_sholm, v3_shol0, v3_head0)),
        udQuad(v3_pabsm, v3_sholm, v3_bedgm, v3_bedg0, v3_shol0)),
        udQuad(v3_pabsm, v3_bedgm, v3_bcenm, v3_bcen0, v3_bedg0)) - zr;
    }

    float distanceHub(vec3 p) {
      // position
      const vec3 p_k  = vec3(+1., -.5, 0.); // King
      const vec3 p_rb = vec3( 0., -.5, 0.); // Rook, Bishop
      const vec3 p_gs = vec3(-1., -.5, 0.); // Gold, Silver
      const vec3 p_n  = vec3(+1., +.5, 0.); // kNight
      const vec3 p_l  = vec3( 0., +.5, 0.); // Lance
      const vec3 p_p  = vec3(-1., +.5, 0.); // Pawn
      // half width/height/thick (square height = 1.)
      const vec3 g_k  = vec3(.371, .413, .125); // King
      const vec3 g_rb = vec3(.358, .400, .120); // Rook, Bishop
      const vec3 g_gs = vec3(.345, .387, .114); // Gold, Silver
      const vec3 g_n  = vec3(.329, .374, .107); // kNight
      const vec3 g_l  = vec3(.303, .362, .103); // Lance
      const vec3 g_p  = vec3(.291, .349, .100); // Pawn
      // rotate
      float sin_x = cos(time * 0.4);
      float cos_x = sin(time * 0.4);
      float sin_z = cos(time * 1.0);
      float cos_z = sin(time * 1.0);
      // rotate matrix
      mat3 m =
        mat3(sin_z, -cos_z, 0., cos_z, sin_z, 0., 0., 0., 1.) *
        mat3(1., 0., 0., 0., sin_x, cos_x, 0., -cos_x, sin_x);
      // distance
      float d[6] = float[](
        sdPiece_simple(m * (p + p_k ), g_k ), // King
        sdPiece_simple(m * (p + p_rb), g_rb), // Rook, Bishop
        sdPiece_simple(m * (p + p_gs), g_gs), // Gold, Silver
        sdPiece_simple(m * (p + p_n ), g_n ), // kNight
        sdPiece_simple(m * (p + p_l ), g_l ), // Lance
        sdPiece_simple(m * (p + p_p ), g_p )  // Pawn
      );
      float d_min = d[0];

      for(int i = 1; i < 6; i++) {
        if(d_min > d[i]) {
          d_min = d[i];
        }
      }

      return d_min;
    }

    vec3 genNormal(vec3 p) {
      const float d = .001;
      const vec3 xp = vec3(+d, 0., 0.);
      const vec3 yp = vec3(0., +d, 0.);
      const vec3 zp = vec3(0., 0., +d);
      const vec3 xn = vec3(-d, 0., 0.);
      const vec3 yn = vec3(0., -d, 0.);
      const vec3 zn = vec3(0., 0., -d);
      return normalize(vec3(
        distanceHub(p + xp) - distanceHub(p + xn),
        distanceHub(p + yp) - distanceHub(p + yn),
        distanceHub(p + zp) - distanceHub(p + zn)
      ));
    }

    void main() {
      const vec3 light = normalize(vec3(-.3, .3, 1.));
      const float err = .001;
      vec2 p = vPosition.xy * resolution.xy / vec2(max(resolution.x, resolution.y));
      vec3 cPos = vec3(0., 0.,  2.0);
      vec3 cDir = vec3(0., 0., -1.0);
      vec3 cUp  = vec3(0., 1.,  0.0);
      vec3 cSide = cross(cDir, cUp);
      float targetDepth = 1.;
      vec3 ray = normalize(cSide * p.x + cUp * p.y + cDir * targetDepth);
      float dist = 10000.;
      float rLen = 0.;
      vec3 rPos = cPos;
      for(int i = 0; i < 64; ++i){
        float prevDist = dist;
        dist = distanceHub(rPos);
        float d = abs(dist);
        if(d < err || (i > 16 && d > 1.)) { break; }
        rLen += dist;
        rPos = cPos + ray * rLen;
      }
      if(abs(dist) < err) {
        vec3 normal = genNormal(rPos);
        float diff = max(dot(normal, light), .2);
        outColor = vec4(vec3(1., .8, .6) * diff, 1.);
      } else {
        outColor = vec4(.2, .6, .3, 1.);
      }
    }
    "#,
  )?;
  let program = link_program(&context, &vert_shader, &frag_shader)?;
  context.use_program(Some(&program));

  let u_time = context.get_uniform_location(&program, "time").unwrap();
  //let u_mouse = context.get_uniform_location(&program, "mouse").unwrap();
  let u_resolution = context.get_uniform_location(&program, "resolution").unwrap();

  let vertices: [f32; 18] = [
    -1.0, -1.0, 0.0,
     1.0, -1.0, 0.0,
     1.0,  1.0, 0.0,
     1.0,  1.0, 0.0,
    -1.0,  1.0, 0.0,
    -1.0, -1.0, 0.0,
  ];

  let buffer = context.create_buffer().ok_or("failed to create buffer")?;
  context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(&buffer));

  // Note that `Float32Array::view` is somewhat dangerous (hence the
  // `unsafe`!). This is creating a raw view into our module's
  // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
  // (aka do a memory allocation in Rust) it'll cause the buffer to change,
  // causing the `Float32Array` to be invalid.
  //
  // As a result, after `Float32Array::view` we have to be very careful not to
  // do any memory allocations before it's dropped.
  unsafe {
    let vert_array = js_sys::Float32Array::view(&vertices);

    context.buffer_data_with_array_buffer_view(
      WebGl2RenderingContext::ARRAY_BUFFER,
      &vert_array,
      WebGl2RenderingContext::STATIC_DRAW,
    );
  }

  context.vertex_attrib_pointer_with_i32(0, 3, WebGl2RenderingContext::FLOAT, false, 0, 0);
  context.enable_vertex_attrib_array(0);

  context.clear_color(0.0, 0.0, 0.0, 1.0);
  context.clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);

  let f = Rc::new(RefCell::new(None));
  let g = f.clone();

  let start_time = js_sys::Date::now();
  let mut i = 0;
  *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
    i += 1;

    let now_time = js_sys::Date::now();
    let duration = (now_time - start_time) as f32 * 0.001;
    /*
    let text = format!("reqAnimFrame time {}", duration);
    p.set_text_content(Some(&text));
    */

    context.uniform1f(Some(&u_time), duration);
    //context.uniform2f(Some(&u_mouse), 0.0 as f32, 0.0 as f32);
    context.uniform2f(Some(&u_resolution), canvas_w as f32, canvas_h as f32);

    context.draw_arrays(
      WebGl2RenderingContext::TRIANGLES,
      0,
      (vertices.len() / 3) as i32,
    );
    context.flush();

    request_animation_frame(f.borrow().as_ref().unwrap());
  }) as Box<dyn FnMut()>));

  request_animation_frame(g.borrow().as_ref().unwrap());

  Ok(())
}

pub fn compile_shader(
  context: &WebGl2RenderingContext,
  shader_type: u32,
  source: &str,
) -> Result<WebGlShader, String> {
  let shader = context
    .create_shader(shader_type)
    .ok_or_else(|| String::from("Unable to create shader object"))?;
  context.shader_source(&shader, source);
  context.compile_shader(&shader);

  if context
    .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
    .as_bool()
    .unwrap_or(false)
  {
    Ok(shader)
  } else {
    Err(context
      .get_shader_info_log(&shader)
      .unwrap_or_else(|| String::from("Unknown error creating shader")))
  }
}

pub fn link_program(
  context: &WebGl2RenderingContext,
  vert_shader: &WebGlShader,
  frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
  let program = context
    .create_program()
    .ok_or_else(|| String::from("Unable to create shader object"))?;

  context.attach_shader(&program, vert_shader);
  context.attach_shader(&program, frag_shader);
  context.link_program(&program);

  if context
    .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
    .as_bool()
    .unwrap_or(false)
  {
      Ok(program)
  } else {
    Err(context
      .get_program_info_log(&program)
      .unwrap_or_else(|| String::from("Unknown error creating program object")))
  }
}
