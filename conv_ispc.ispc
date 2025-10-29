// ================================================================
// conv_ispc.ispc - ISPC并行卷积实现（修正版）
// 修复嵌套foreach错误：仅保留最外层foreach并用for循环代替内层
// ================================================================

export void ispc_conv1(
    uniform float input_data[],       // input feature map (flattened)
    uniform int input_channels,
    uniform int input_width,
    uniform int input_height,
    uniform float kernel_weights[],   // kernel weights (flattened)
    uniform int kernel_size,
    uniform int kernel_in_channels,
    uniform int kernel_out_channels,
    uniform float bias_values[],      // biases (flattened)
    uniform float output_data[],      // output feature map (flattened)
    uniform int output_width,
    uniform int output_height
) {
    // 仅对输出通道并行
    foreach (out_c = 0 ... kernel_out_channels) {
        for (uniform int out_h = 0; out_h < output_height; out_h++) {
            for (uniform int out_w = 0; out_w < output_width; out_w++) {

                float psum = 0.0f;

                for (uniform int k_h = 0; k_h < kernel_size; k_h++) {
                    for (uniform int k_w = 0; k_w < kernel_size; k_w++) {
                        for (uniform int in_c = 0; in_c < input_channels; in_c++) {

                            int in_h = out_h + k_h;
                            int in_w = out_w + k_w;

                            int input_idx = in_c * input_height * input_width
                                            + in_h * input_width + in_w;

                            int kernel_idx = out_c * kernel_in_channels * kernel_size * kernel_size
                                            + in_c * kernel_size * kernel_size
                                            + k_h * kernel_size + k_w;

                            psum += input_data[input_idx] * kernel_weights[kernel_idx];
                        }
                    }
                }

                int output_idx = out_c * output_height * output_width
                                 + out_h * output_width + out_w;
                output_data[output_idx] = psum + bias_values[out_c];
            }
        }
    }
}


// ================================================================
// 第二个卷积函数同理
// ================================================================

export void ispc_conv2(
    uniform float input_data[],       // input feature map (flattened)
    uniform int input_channels,
    uniform int input_width,
    uniform int input_height,
    uniform float kernel_weights[],   // kernel weights (flattened)
    uniform int kernel_size,
    uniform int kernel_in_channels,
    uniform int kernel_out_channels,
    uniform float bias_values[],      // biases (flattened)
    uniform float output_data[],      // output feature map (flattened)
    uniform int output_width,
    uniform int output_height
) {
    // 外层 foreach 并行通道方向
    foreach (out_c = 0 ... kernel_out_channels) {
        for (uniform int out_h = 0; out_h < output_height; out_h++) {
            for (uniform int out_w = 0; out_w < output_width; out_w++) {

                float psum = 0.0f;

                for (uniform int k_h = 0; k_h < kernel_size; k_h++) {
                    for (uniform int k_w = 0; k_w < kernel_size; k_w++) {
                        for (uniform int in_c = 0; in_c < input_channels; in_c++) {

                            int in_h = out_h + k_h;
                            int in_w = out_w + k_w;

                            int input_idx = in_c * input_height * input_width
                                            + in_h * input_width + in_w;

                            int kernel_idx = out_c * kernel_in_channels * kernel_size * kernel_size
                                            + in_c * kernel_size * kernel_size
                                            + k_h * kernel_size + k_w;

                            psum += input_data[input_idx] * kernel_weights[kernel_idx];
                        }
                    }
                }

                int output_idx = out_c * output_height * output_width
                                 + out_h * output_width + out_w;
                output_data[output_idx] = psum + bias_values[out_c];
            }
        }
    }
}
