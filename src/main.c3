import std::io;
import mlx::mx;

def Void = void;
def Float32 = float;
def Int32 = int;
def Char = char;

fn Void print_array(String msg, mx::Array arr) {
   mx::String str = mx::tostring(arr);
   defer mx::free(str);

   @pool() {
      io::printfn("%s", msg);
      io::printfn("%s", ((ZString) mx::string_data(str)).copy());
   };
}  

fn Void example() {
   mx::Stream stream = mx::gpu_stream();
   defer mx::free(stream);

   Float32[6] data = {1, 2, 3, 4, 5, 6};
   Int32[2] shape = {2, 3};

   mx::Array arr = mx::array_from_data(&data, &shape, 2, mx::FLOAT32);
   defer mx::free(arr);

   print_array("hello world!", arr);

   mx::Array two = mx::array_from_int(2); 
   defer mx::free(two);

   mx::Array res = mx::divide(arr, two, stream);
   defer mx::free(res);

   mx::Array arange = mx::arange(0, 3, 0.5, mx::FLOAT32, stream);
   defer mx::free(arange);
}

fn mx::Array inc(mx::Array input) {
   mx::Stream stream = mx::gpu_stream();
   defer mx::free(stream);

   mx::Array y = mx::array_from_float(1.0f);
   defer mx::free(y);

   mx::Array res = mx::add(input, y, stream);
   return res;
}

fn mx::Array inc_value(mx::Array input, Void* payload) {
   mx::Stream stream = mx::gpu_stream();
   defer mx::free(stream);

   if (mx::vector_array_size(input) != 1) {
      io::eprint("inc_func_value: expected 1 argument");
      return input;
   }

   mx::Array value = (mx::Array) payload;
   mx::Array res = mx::add(mx::vector_array_get(input, 0), value, stream);
   defer mx::free(res);
   mx::VectorArray vres = mx::vector_array_from_array(res);
   return vres;
}

fn Void example_grad() {
   mx::Array x = mx::array_from_float(1.0f); defer mx::free(x);
   mx::Array y = mx::array_from_float(1.0f); defer mx::free(y);

   mx::Closure cls = mx::closure_new_unary(&inc);
   defer mx::free(cls);
   mx::Closure cls_with_value = 
      mx::closure_new_with_payload(&inc_value, y, &mx::free);
   defer mx::free(cls_with_value);

   {| io::printn("jvp:");

      mx::Array one = mx::array_from_float(1.0); 
      defer mx::free(one);

      mx::VectorArray primals = mx::vector_array_from_array(x); 
      defer mx::free(primals);

      mx::VectorArray tangents = mx::vector_array_from_array(one);
      defer mx::free(tangents);

      mx::VectorVectorArray res = mx::jvp(cls, primals, tangents);
      defer mx::free(res);

      mx::Array out = mx::vector_vector_array_get2d(res, 0, 0);
      defer mx::free(out);

      mx::Array dout = mx::vector_vector_array_get2d(res, 1, 0);
      defer mx::free(dout);

      print_array("out", out);
      print_array("dout", dout);
   |};

   {| io::printn("value_and_grad");

      Int32 garg = 0i32;

      mx::ClosureValueAndGrad vag = mx::value_and_grad(cls, &garg, 1); 
      defer mx::free(vag);

      mx::VectorArray inputs = mx::vector_array_from_array(x);
      defer mx::free(inputs);

      mx::VectorVectorArray res = mx::closure_value_and_grad_apply(vag, inputs);
      defer mx::free(res);

      mx::Array out = mx::vector_vector_array_get2d(res, 0, 0);
      defer mx::free(out);

      mx::Array dout = mx::vector_vector_array_get2d(res, 1, 0);
      defer mx::free(dout);
      
      print_array("out", out);
      print_array("dout", dout);
   |};

   {| io::printn("value_and_grad with payload");

      Int32 garg = 0i32;

      mx::ClosureValueAndGrad vag = mx::value_and_grad(cls_with_value, &garg, 1);
      defer mx::free(vag);

      mx::VectorArray inputs = mx::vector_array_from_array(x);
      defer mx::free(inputs);

      mx::VectorVectorArray res = mx::closure_value_and_grad_apply(vag, inputs);
      defer mx::free(res);

      mx::Array out = mx::vector_vector_array_get2d(res, 0, 0);
      defer mx::free(out);

      mx::Array dout = mx::vector_vector_array_get2d(res, 1, 0);
      defer mx::free(dout);
      
      print_array("out", out);
      print_array("dout", dout);
   |};
}

fn Void example_safetensors() {   
   File! openning = io::file::open("arrays.safetensors", "r"); 
   if (catch err = openning) {
      io::eprintn("could not load <arrays.safetensors>");
      return;
   } 

   File file = openning; 
   defer { if (catch err = file.close()) io::eprintfn("Fault closing: %s", err);}

   mx::Stream stream = mx::gpu_stream(); 
   defer mx::free(stream);

   mx::SafeTensors st = mx::load_safetensors_file(file.file, stream);
   defer mx::free(st);

   mx::MapStringToArray map = mx::safetensors_data(st);
   defer mx::free(map);

   mx::MapStringToArrayIterator it = mx::map_string_to_array_iterate(map);
   defer mx::free(it);

   while (!mx::map_string_to_array_iterator_end(it)) {
      @pool() { 
         mx::String key = mx::map_string_to_array_iterator_key(it);
         defer mx::free(key);

         mx::Array value = mx::map_string_to_array_iterator_value(it);
         defer mx::free(value);

         print_array(((ZString) mx::string_data(key)).copy(), value);

         mx::map_string_to_array_iterator_next(it);
      };
   }
}

fn Int32 main() {  
   io::printn("[Example]"); example();
   io::printn("[Example-Grad]"); example_grad();
   io::printn("[Example-Safetensors]"); example_safetensors();
   return 0i32;  
}
