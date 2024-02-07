use dam::context_tools::*;

pub fn ctx_print<T: Context, S: AsRef<str>>(name: S, ctx: T) -> T {
    println!("Registering {} with ID {:?}", name.as_ref(), ctx.id());
    ctx
}
