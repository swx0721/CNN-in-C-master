#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <windows.h>
#include "conv_ispc.h"
// 如果希望重置模型训练进度，直接删去本目录下的 kernels, weights, biases 三个文件即可
#define RED "\x1B[31m"
#define RESET "\x1B[0m"

#define INPUT_DIMENSIONS 28
#define C1_LENGTH 6
#define C1_DIMENSIONS 24
#define S1_LENGTH 6
#define S1_DIMENSIONS 12
#define C2_LENGTH 12
#define C2_DIMENSIONS 8
#define S2_LENGTH 12
#define S2_DIMENSIONS 4
#define OUTPUT_LENGTH 10
#define KERNEL_SIZE 5
#define UPPER 1
#define LOWER -1
#define BATCH_SIZE 200
#define LEARNING_RATE 0.01
// 将训练集规模缩小为 600（可按需调整）
#define TRAINING_SET_SIZE 600
#define USE_ISPC 0 // 串行、并行的开关宏，其中：1 = 并行模式，0 = 串行模式

// 增加卷积层计时统计变量
double conv1_total_time = 0.0;
double conv2_total_time = 0.0;
int conv1_calls = 0;
int conv2_calls = 0;

struct VECTOR
{
    double *vector;
};
typedef struct VECTOR vector;

vector *new_vector(int length)
{
    vector *new = malloc(sizeof(vector));
    new->vector = malloc(sizeof(double) * length);
    return new;
}

struct ARRAY
{
    double **matrix;
};
typedef struct ARRAY array;

array *new_array(int rows, int columns)
{
    array *new = malloc(sizeof(array *));
    new->matrix = malloc(sizeof(double *) * rows);
    int i;
    for (i = 0; i < rows; i++)
    {
        new->matrix[i] = malloc(sizeof(double) * columns);
    }
    return new;
}

struct KERNEL_ARRAY
{
    array ***kernels;
};
typedef struct KERNEL_ARRAY kernel_array;

kernel_array *new_kernel_array(int rows, int columns, int kernel_dimensions)
{
    kernel_array *new = malloc(sizeof(kernel_array *));
    new->kernels = malloc(sizeof(array **) * rows);
    int j, k;
    for (j = 0; j < rows; j++)
    {
        new->kernels[j] = malloc(sizeof(array *) * columns);
    }
    for (j = 0; j < rows; j++)
    {
        for (k = 0; k < columns; k++)
        {
            new->kernels[j][k] = new_array(kernel_dimensions, kernel_dimensions);
        }
    }
    return new;
}

struct IMAGE_VECTOR
{
    array **image;
};
typedef struct IMAGE_VECTOR image_vector;

image_vector *new_image_vector(int image_dimensions, int length)
{
    image_vector *new = malloc(sizeof(image_vector *));
    new->image = malloc(sizeof(array *) * length);
    int i;
    for (i = 0; i < length; i++)
    {
        new->image[i] = new_array(image_dimensions, image_dimensions);
    }
    return new;
}

struct CNN
{
    array *input_image;
    kernel_array *C1_Kernels;
    vector *C1_Biases;
    image_vector *C1_Images;
    image_vector *S1_Images;
    kernel_array *C2_Kernels;
    vector *C2_Biases;
    image_vector *C2_Images;
    image_vector *S2_Images;
    vector *S2_Vector;
    array *output_weights;
    vector *output_biases;
    vector *calculated_output;
    vector *desired_output;
};
typedef struct CNN cnn;

cnn *new_cnn()
{
    /* 分配 cnn 结构体的正确大小，并初始化成员为 NULL，随后使用构造函数创建具体对象 */
    cnn *new = malloc(sizeof(cnn));
    if (!new)
    {
        fprintf(stderr, "Failed to allocate memory for cnn struct\n");
        return NULL;
    }
    new->input_image = NULL;
    new->C1_Kernels = NULL;
    new->C1_Biases = NULL;
    new->C1_Images = NULL;
    new->S1_Images = NULL;
    new->C2_Kernels = NULL;
    new->C2_Biases = NULL;
    new->C2_Images = NULL;
    new->S2_Images = NULL;
    new->S2_Vector = NULL;
    new->output_weights = NULL;
    new->output_biases = NULL;
    new->calculated_output = NULL;
    new->desired_output = NULL;
    new->input_image = new_array(INPUT_DIMENSIONS, INPUT_DIMENSIONS);
    new->C1_Kernels = new_kernel_array(1, C1_LENGTH, KERNEL_SIZE);
    new->C1_Biases = new_vector(C1_LENGTH);
    new->C1_Images = new_image_vector(C1_DIMENSIONS, C1_LENGTH);
    new->S1_Images = new_image_vector(S1_DIMENSIONS, S1_LENGTH);
    new->C2_Kernels = new_kernel_array(S1_LENGTH, C2_LENGTH, KERNEL_SIZE);
    new->C2_Biases = new_vector(C2_LENGTH);
    new->C2_Images = new_image_vector(C2_DIMENSIONS, C2_LENGTH);
    new->S2_Images = new_image_vector(S2_DIMENSIONS, S2_LENGTH);
    new->S2_Vector = new_vector(S2_DIMENSIONS * S2_DIMENSIONS * S2_LENGTH);
    new->output_weights = new_array(OUTPUT_LENGTH, S2_DIMENSIONS * S2_DIMENSIONS * S2_LENGTH);
    new->output_biases = new_vector(OUTPUT_LENGTH);
    new->calculated_output = new_vector(OUTPUT_LENGTH);
    new->desired_output = new_vector(OUTPUT_LENGTH);
    return new;
}

void load_kernels(cnn *network);
void load_network(cnn *network);
void load_weights(cnn *network);
void load_biases(cnn *network);
void load_output(cnn *network);
double uniform_rand();
void zero_activations(cnn *network);
void zero_parameters(cnn *network);
void zero_network(cnn *network);
void read_headers(FILE *image_ptr, FILE *label_ptr);
void load_image(cnn *network, FILE *image_ptr, FILE *label_ptr);
void load_C1(cnn *network);
void load_S1(cnn *network);
void load_C2(cnn *network);
void load_S2(cnn *network);
void concatenate_S2(cnn *network);
void load_output(cnn *network);
double dot_product(array *array1, array *array2, int dimensions);
double activation(double x);
double activation_derivative(double x);
void free_array(array *array, int rows);
void free_image_vector(image_vector *images, int dimensions, int length);
double average_matrix(array *image, int dimensions);
void backpropagation(cnn *network, cnn *gradient);
void forward_propagate(cnn *network);
void gradient_descent(cnn *network, cnn *gradient, int actual_batch_size);
double loss_function(cnn *network);
void print_output(cnn *network);
void update_batch_gradient(cnn *image_gradient, cnn *batch_gradient);
void print_image(array *image, int dimensions);
double output_activation(double x);
double output_activation_derivative(double x);
void save_parameters(cnn *network);
void save_kernels(cnn *network);
void save_weights(cnn *network);
void save_biases(cnn *network);

int runtime_use_ispc = 0;

int main(int argc, char **argv)
{
    // 设置控制台编码
    SetConsoleOutputCP(65001);
    FILE *image_ptr;
    FILE *label_ptr;
    srand(time(NULL));
    printf("main() started\n");
    fflush(stdout);
    // 运行时可配置项（默认使用宏定义）
    int run_train_size = TRAINING_SET_SIZE;
    int disable_load = 0;
    int disable_save = 0;
    // 解析命令行参数
    for (int ai = 1; ai < argc; ai++)
    {
        if (strcmp(argv[ai], "--no-load") == 0)
            disable_load = 1;
        else if (strcmp(argv[ai], "--no-save") == 0)
            disable_save = 1;
        else if (strcmp(argv[ai], "--train-size") == 0 && ai + 1 < argc)
        {
            run_train_size = atoi(argv[++ai]);
            if (run_train_size <= 0)
                run_train_size = TRAINING_SET_SIZE;
        }
    }
    cnn *network = new_cnn();
    printf("created network\n");
    fflush(stdout);
    cnn *batch_gradient = new_cnn();
    printf("created batch_gradient\n");
    fflush(stdout);
    cnn *image_gradient = new_cnn();
    printf("created image_gradient\n");
    fflush(stdout);

    zero_network(batch_gradient);
    printf("zeroed batch_gradient\n");
    fflush(stdout);
    zero_network(image_gradient);
    printf("zeroed image_gradient\n");
    fflush(stdout);
    zero_network(network);
    printf("zeroed network\n");
    fflush(stdout);
    printf("After zero_network, before load_network\n");
    fflush(stdout);
    if (!disable_load)
    {
        load_network(network);
    }
    else
    {
        printf("Skipping load_network (fresh start)\n");
        fflush(stdout);
    }
    printf("After load_network\n");
    fflush(stdout);
    int n, i, j, m;

    // 我们将做两次训练：一次串行（runtime_use_ispc=0），一次并行（runtime_use_ispc=1）
    int mode;
    double serial_elapsed = 0.0, parallel_elapsed = 0.0;
    double serial_conv1 = 0.0, serial_conv2 = 0.0, parallel_conv1 = 0.0, parallel_conv2 = 0.0;

    for (mode = 0; mode < 2; mode++)
    {
        // 0 -> 串行，1 -> 并行
        int use_ispc_runtime_mode = mode; // 临时标志，传递给运行时变量
        // 将运行时开关赋给全局 runtime_use_ispc（在文件顶部新增 int runtime_use_ispc; 若尚未，添加）
        runtime_use_ispc = use_ispc_runtime_mode;

        // 每次运行都从头开始，不加载上次训练参数，且不保存中间参数以免影响时间统计
        int local_disable_load = 1;
        int local_disable_save = 1;

        // 清零网络相关累加器和梯度
        zero_network(batch_gradient);
        zero_network(image_gradient);
        zero_network(network);

        // 清空卷积计时计数，开始本次计时
        conv1_total_time = 0.0;
        conv2_total_time = 0.0;
        conv1_calls = 0;
        conv2_calls = 0;

        clock_t train_start_local = clock();

        // 训练循环（与原逻辑一致）
        double batch_loss_local = 0;
        double iteration_loss_local = 0;
        for (m = 0; m < 200; m++)
        {
            image_ptr = fopen("train-images.idx3-ubyte", "rb");
            label_ptr = fopen("train-labels.idx1-ubyte", "rb");
            if (!image_ptr)
            {
                fprintf(stderr, "Failed to open training images file 'train-images.idx3-ubyte'\n");
                return EXIT_FAILURE;
            }
            if (!label_ptr)
            {
                fprintf(stderr, "Failed to open training labels file 'train-labels.idx1-ubyte'\n");
                fclose(image_ptr);
                return EXIT_FAILURE;
            }
            printf("Mode %s: Opened training files for iteration %d\n", runtime_use_ispc ? "PARALLEL" : "SERIAL", m);
            fflush(stdout);
            read_headers(image_ptr, label_ptr);

            int batches_per_iter = (run_train_size + BATCH_SIZE - 1) / BATCH_SIZE;
            for (n = 0; n < batches_per_iter; n++)
            {
                int current_batch_size = BATCH_SIZE;
                if (n == batches_per_iter - 1 && (run_train_size % BATCH_SIZE) != 0)
                    current_batch_size = run_train_size % BATCH_SIZE;

                zero_network(batch_gradient);
                for (i = 0; i < current_batch_size; i++)
                {
                    zero_network(image_gradient);
                    zero_activations(network);
                    load_image(network, image_ptr, label_ptr);
                    forward_propagate(network);
                    backpropagation(network, image_gradient);
                    update_batch_gradient(image_gradient, batch_gradient);
                    batch_loss_local += loss_function(network);
                }

                gradient_descent(network, batch_gradient, current_batch_size);
                printf("Mode %s: Loss for batch(%d, %d): %lf\n", runtime_use_ispc ? "PAR" : "SER", m, n, batch_loss_local / (double)current_batch_size);
                fflush(stdout);
                iteration_loss_local += batch_loss_local;
                batch_loss_local = 0;
                // 不保存以免 IO 影响时间统计
            }

            printf("Mode %s: Loss for iteration(%d): %lf\n", runtime_use_ispc ? "PARALLEL" : "SERIAL", m, iteration_loss_local / (double)run_train_size);
            fflush(stdout);
            iteration_loss_local = 0;
            fclose(image_ptr);
            fclose(label_ptr);
        } // end epochs

        clock_t train_end_local = clock();
        double elapsed_local = (double)(train_end_local - train_start_local) / (double)CLOCKS_PER_SEC;

        // 保存本次运行结果
        if (mode == 0)
        {
            serial_elapsed = elapsed_local;
            serial_conv1 = conv1_total_time;
            serial_conv2 = conv2_total_time;
        }
        else
        {
            parallel_elapsed = elapsed_local;
            parallel_conv1 = conv1_total_time;
            parallel_conv2 = conv2_total_time;
        }

        printf("\nMode %s finished. Elapsed: %.3f s, conv1 total: %.6f s (%d calls), conv2 total: %.6f s (%d calls)\n\n",
               runtime_use_ispc ? "PARALLEL" : "SERIAL",
               elapsed_local,
               conv1_total_time, conv1_calls,
               conv2_total_time, conv2_calls);
        fflush(stdout);
    } // end mode loop

    // 训练两次后，打印对比汇总（串行、并行）
    printf("\n=== 两种模式耗时对比 ===\n");
    printf("串行总训练时间: %.3f 秒\n", serial_elapsed);
    printf("并行总训练时间: %.3f 秒\n", parallel_elapsed);
    printf("串行 C1 总耗时: %.6f s, C2 总耗时: %.6f s\n", serial_conv1, serial_conv2);
    printf("并行 C1 总耗时: %.6f s, C2 总耗时: %.6f s\n", parallel_conv1, parallel_conv2);
    if (parallel_elapsed > 0.0)
        printf("总体加速比 (串行/并行): %.3f\n", serial_elapsed / parallel_elapsed);
    fflush(stdout);

    // 防止双击运行时窗口马上关闭，等待回车
    printf("\n按回车键退出程序...\n");
    fflush(stdout);
    getchar();

    return (EXIT_SUCCESS);
}

//--------------------------主函数结束----------------------------------

// 新增代码部分
// 卷积输入一维化
float *flatten_conv1_input(float conv1_input[600][1][6][6], int img_idx)
{
    return (float *)conv1_input[img_idx];
}
// 卷积核一维化
float *flatten_conv1_kernel(float conv1_kernel[6][1][3][3])
{
    return (float *)conv1_kernel;
}
// 输出一维化
float *flatten_conv1_output(float conv1_output[600][6][6][6], int img_idx)
{
    return (float *)conv1_output[img_idx];
}

//--------------------------结束--------------------------
void save_parameters(cnn *network)
{
    save_kernels(network);
    save_weights(network);
    save_biases(network);
}

void save_kernels(cnn *network)
{
    FILE *kernels_ptr = fopen("kernels", "w");
    int i, n, j, k;

    for (i = 0; i < C1_LENGTH; i++)
    {
        for (j = 0; j < KERNEL_SIZE; j++)
        {
            for (k = 0; k < KERNEL_SIZE; k++)
            {
                fprintf(kernels_ptr, "%lf ", network->C1_Kernels->kernels[0][i]->matrix[j][k]);
            }
        }
    }

    for (i = 0; i < S1_LENGTH; i++)
    {
        for (n = 0; n < C2_LENGTH; n++)
        {
            for (j = 0; j < KERNEL_SIZE; j++)
            {
                for (k = 0; k < KERNEL_SIZE; k++)
                {
                    fprintf(kernels_ptr, "%lf ", network->C2_Kernels->kernels[i][n]->matrix[j][k]);
                }
            }
        }
    }
    fclose(kernels_ptr);
}

void save_weights(cnn *network)
{
    FILE *weight_ptr = fopen("weights", "w");
    int j, k;
    for (j = 0; j < OUTPUT_LENGTH; j++)
    {
        for (k = 0; k < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; k++)
        {
            fprintf(weight_ptr, "%lf ", network->output_weights->matrix[j][k]);
        }
    }
    fclose(weight_ptr);
}

void save_biases(cnn *network)
{
    FILE *bias_ptr = fopen("biases", "w");
    int i;
    for (i = 0; i < C1_LENGTH; i++)
    {
        fprintf(bias_ptr, "%lf ", network->C1_Biases->vector[i]);
    }
    for (i = 0; i < C2_LENGTH; i++)
    {
        fprintf(bias_ptr, "%lf ", network->C2_Biases->vector[i]);
    }
    for (i = 0; i < OUTPUT_LENGTH; i++)
    {
        fprintf(bias_ptr, "%lf ", network->output_biases->vector[i]);
    }
    fclose(bias_ptr);
}

void print_output(cnn *network)
{
    int i;
    for (i = 0; i < OUTPUT_LENGTH; i++)
    {
        printf("%lf == %lf\n", network->calculated_output->vector[i], network->desired_output->vector[i]);
    }
    printf("\n");
}

void forward_propagate(cnn *network)
{
    load_C1(network);
    load_S1(network);
    load_C2(network);
    load_S2(network);
    concatenate_S2(network);
    load_output(network);
}

// Loads the network's weights, kernels, and biases
void load_network(cnn *network)
{
    load_kernels(network);
    load_weights(network);
    load_biases(network);
}

// Loads a uniform distribution of values for the kernels in the network
void load_kernels(cnn *network)
{
    int i, n, j, k;
    FILE *kernel_ptr = fopen("kernels", "r");
    if (!kernel_ptr)
    {
        printf("kernels file not found, will use initialized parameters (skipping load_kernels)\n");
        fflush(stdout);
        return;
    }
    double value = 0;
    // C1 will have C1_LENGTH, KERNEL_SIZE x KERNEL_SIZE kernels
    for (i = 0; i < C1_LENGTH; i++)
    {
        for (j = 0; j < KERNEL_SIZE; j++)
        {
            for (k = 0; k < KERNEL_SIZE; k++)
            {
                fscanf(kernel_ptr, "%lf", &value);
                network->C1_Kernels->kernels[0][i]->matrix[j][k] = value;
            }
        }
        // print_image(network->C1_Kernels->kernels[0][i], KERNEL_SIZE);
    }

    // C2 will have C2_LENGTH, KERNEL_SIZE x KERNEL_SIZE kernels
    for (i = 0; i < S1_LENGTH; i++)
    {
        for (n = 0; n < C2_LENGTH; n++)
        {
            for (j = 0; j < KERNEL_SIZE; j++)
            {
                for (k = 0; k < KERNEL_SIZE; k++)
                {
                    fscanf(kernel_ptr, "%lf", &value);
                    network->C2_Kernels->kernels[i][n]->matrix[j][k] = value;
                }
            }
            // print_image(network->C2_Kernels->kernels[i][n], KERNEL_SIZE);
        }
    }
    fclose(kernel_ptr);
}

// Loads a uniform distribution of values for the weights in the network
void load_weights(cnn *network)
{
    FILE *weight_ptr = fopen("weights", "r");
    if (!weight_ptr)
    {
        printf("weights file not found, will use initialized parameters (skipping load_weights)\n");
        fflush(stdout);
        return;
    }
    int j, k;
    double value = 0;
    for (j = 0; j < OUTPUT_LENGTH; j++)
    {
        for (k = 0; k < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; k++)
        {
            fscanf(weight_ptr, "%lf", &value);
            network->output_weights->matrix[j][k] = value;
        }
    }
    fclose(weight_ptr);
}

// Sets all biases to 0
void load_biases(cnn *network)
{
    int i;
    FILE *biases_ptr = fopen("biases", "r");
    if (!biases_ptr)
    {
        printf("biases file not found, will use initialized parameters (skipping load_biases)\n");
        fflush(stdout);
        return;
    }
    double value = 0;
    for (i = 0; i < C1_LENGTH; i++)
    {
        fscanf(biases_ptr, "%lf", &value);
        network->C1_Biases->vector[i] = value;
    }
    for (i = 0; i < C2_LENGTH; i++)
    {
        fscanf(biases_ptr, "%lf", &value);
        network->C2_Biases->vector[i] = value;
    }
    for (i = 0; i < OUTPUT_LENGTH; i++)
    {
        fscanf(biases_ptr, "%lf", &value);
        network->output_biases->vector[i] = value;
    }
    fclose(biases_ptr);
}

// Sets all values within a network to 0
void zero_network(cnn *network)
{
    zero_activations(network);
    zero_parameters(network);
}

// Sets all activations within a network to 0
void zero_activations(cnn *network)
{
    int i, j, k;
    for (i = 0; i < C1_LENGTH; i++)
    {
        for (j = 0; j < C1_DIMENSIONS; j++)
        {
            for (k = 0; k < C1_DIMENSIONS; k++)
            {
                network->C1_Images->image[i]->matrix[j][k] = 0;
            }
        }
    }
    for (i = 0; i < S1_LENGTH; i++)
    {
        for (j = 0; j < S1_DIMENSIONS; j++)
        {
            for (k = 0; k < S1_DIMENSIONS; k++)
            {
                network->S1_Images->image[i]->matrix[j][k] = 0;
            }
        }
    }
    for (i = 0; i < C2_LENGTH; i++)
    {
        for (j = 0; j < C2_DIMENSIONS; j++)
        {
            for (k = 0; k < C2_DIMENSIONS; k++)
            {
                network->C2_Images->image[i]->matrix[j][k] = 0;
            }
        }
    }
    for (i = 0; i < S2_LENGTH; i++)
    {
        for (j = 0; j < S2_DIMENSIONS; j++)
        {
            for (k = 0; k < S2_DIMENSIONS; k++)
            {
                network->S2_Images->image[i]->matrix[j][k] = 0;
            }
        }
    }
    for (i = 0; i < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; i++)
    {
        network->S2_Vector->vector[i] = 0;
    }
    for (i = 0; i < OUTPUT_LENGTH; i++)
    {
        network->calculated_output->vector[i] = 0;
    }
    for (i = 0; i < OUTPUT_LENGTH; i++)
    {
        network->desired_output->vector[i] = 0;
    }
}

// Sets all parameters within a network to 0
void zero_parameters(cnn *network)
{
    int i, n, j, k;

    for (i = 0; i < C1_LENGTH; i++)
    {
        for (j = 0; j < KERNEL_SIZE; j++)
        {
            for (k = 0; k < KERNEL_SIZE; k++)
            {
                network->C1_Kernels->kernels[0][i]->matrix[j][k] = 0;
            }
        }
    }
    for (i = 0; i < S1_LENGTH; i++)
    {
        for (n = 0; n < C2_LENGTH; n++)
        {
            for (j = 0; j < KERNEL_SIZE; j++)
            {
                for (k = 0; k < KERNEL_SIZE; k++)
                {
                    network->C2_Kernels->kernels[i][n]->matrix[j][k] = 0;
                }
            }
        }
    }
    for (j = 0; j < OUTPUT_LENGTH; j++)
    {
        for (k = 0; k < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; k++)
        {
            network->output_weights->matrix[j][k] = 0;
        }
    }
    for (i = 0; i < C1_LENGTH; i++)
    {
        network->C1_Biases->vector[i] = 0;
    }
    for (i = 0; i < C2_LENGTH; i++)
    {
        network->C2_Biases->vector[i] = 0;
    }
    for (i = 0; i < OUTPUT_LENGTH; i++)
    {
        network->output_biases->vector[i] = 0;
    }
}

// Generates a uniform random digit within a specified range
double uniform_rand()
{
    double value = (double)rand() / RAND_MAX;
    value *= (UPPER - LOWER);
    value += LOWER;
    return value;
}

// Reads header items from input files
void read_headers(FILE *image_ptr, FILE *label_ptr)
{
    uint32_t label_magic_number = 0u;
    label_magic_number |= getc(label_ptr) << 24;
    label_magic_number |= getc(label_ptr) << 16;
    label_magic_number |= getc(label_ptr) << 8;
    label_magic_number |= getc(label_ptr);

    // Item count from label file
    uint32_t label_item_count = 0u;
    label_item_count |= getc(label_ptr) << 24;
    label_item_count |= getc(label_ptr) << 16;
    label_item_count |= getc(label_ptr) << 8;
    label_item_count |= getc(label_ptr);

    // Magic Number from images file
    uint32_t images_magic_number = 0u;
    images_magic_number |= getc(image_ptr) << 24;
    images_magic_number |= getc(image_ptr) << 16;
    images_magic_number |= getc(image_ptr) << 8;
    images_magic_number |= getc(image_ptr);

    // Item count from images file
    uint32_t images_item_count = 0u;
    images_item_count |= getc(image_ptr) << 24;
    images_item_count |= getc(image_ptr) << 16;
    images_item_count |= getc(image_ptr) << 8;
    images_item_count |= getc(image_ptr);

    // Rows per image
    uint32_t rows = 0u;
    rows |= getc(image_ptr) << 24;
    rows |= getc(image_ptr) << 16;
    rows |= getc(image_ptr) << 8;
    rows |= getc(image_ptr);

    // Columns per image
    uint32_t columns = 0u;
    columns |= getc(image_ptr) << 24;
    columns |= getc(image_ptr) << 16;
    columns |= getc(image_ptr) << 8;
    columns |= getc(image_ptr);
}

// Reads in an image and it's label from the the input files
void load_image(cnn *network, FILE *image_ptr, FILE *label_ptr)
{
    int j, k;
    double value = 0;
    for (j = 0; j < INPUT_DIMENSIONS; j++)
    {
        for (k = 0; k < INPUT_DIMENSIONS; k++)
        {
            value = (double)getc(image_ptr);
            value /= 255;
            network->input_image->matrix[j][k] = value;
        }
    }
    double t = (double)getc(label_ptr);
    for (j = 0; j < 10; j++)
    {
        if ((int)t == j)
            network->desired_output->vector[j] = 1;
        else
            network->desired_output->vector[j] = 0;
        // printf("%lf\n",network->desired_output->vector[j]);
    }
    // print_image(network->input_image, 28);
}

// Propagate from input to C1
void load_C1(cnn *network)
{
    clock_t _t_start = clock(); // conv1 开始计时

#if USE_ISPC == 1
    // ISPC 并行分支
    int in_w = INPUT_DIMENSIONS;
    int in_h = INPUT_DIMENSIONS;
    int in_ch = 1;                   // 单通道输入
    int out_ch = C1_LENGTH;
    int ksize = KERNEL_SIZE;
    int out_w = C1_DIMENSIONS;
    int out_h = C1_DIMENSIONS;

    // 分配并填充 input (单张图，单通道)
    float *input_f = (float *)malloc(sizeof(float) * in_w * in_h);
    for (int y = 0; y < in_h; y++)
        for (int x = 0; x < in_w; x++)
            input_f[y * in_w + x] = (float)network->input_image->matrix[y][x];

    // 分配并填充 kernels: layout [out_ch][ksize][ksize]
    float *kernels_f = (float *)malloc(sizeof(float) * out_ch * ksize * ksize);
    for (int oc = 0; oc < out_ch; oc++)
    {
        for (int j = 0; j < ksize; j++)
        {
            for (int k = 0; k < ksize; k++)
            {
                kernels_f[(oc * ksize + j) * ksize + k] = 
                    (float)network->C1_Kernels->kernels[0][oc]->matrix[j][k];
            }
        }
    }

    // biases -> float
    float *bias_f = (float *)malloc(sizeof(float) * out_ch);
    for (int oc = 0; oc < out_ch; oc++)
        bias_f[oc] = (float)network->C1_Biases->vector[oc];

    // 输出缓冲
    float *output_f = (float *)malloc(sizeof(float) * out_ch * out_w * out_h);

    // 直接包含头文件中声明的函数
    #include "conv_ispc.h"
    // 调用 ISPC 函数，参数按 conv_ispc.h 的定义
    ispc_conv1(
        input_f, 
        in_ch, in_w, in_h,
        kernels_f, ksize, in_ch, out_ch,
        bias_f,
        output_f,
        out_w, out_h
    );

    // 拷回 network->C1_Images 并应用激活
    for (int oc = 0; oc < out_ch; oc++)
        for (int y = 0; y < out_h; y++)
            for (int x = 0; x < out_w; x++)
            {
                double v = (double)output_f[(oc * out_h + y) * out_w + x];
                network->C1_Images->image[oc]->matrix[y][x] = activation(v);
            }

    free(input_f);
    free(kernels_f);
    free(bias_f);
    free(output_f);

#else
    // 串行分支保持原样
    int length = C1_DIMENSIONS * C1_DIMENSIONS;
    image_vector *input_sections = new_image_vector(KERNEL_SIZE, length);
    int i, j, k;
    for (i = 0; i < length; i++)
        for (j = 0; j < KERNEL_SIZE; j++)
            for (k = 0; k < KERNEL_SIZE; k++)
                input_sections->image[i]->matrix[j][k] =
                    network->input_image->matrix[(int)floor((double)i / C1_DIMENSIONS) + j][i % C1_DIMENSIONS + k];

    for (i = 0; i < C1_LENGTH; i++)
        for (j = 0; j < C1_DIMENSIONS; j++)
            for (k = 0; k < C1_DIMENSIONS; k++)
                network->C1_Images->image[i]->matrix[j][k] = activation(
                    dot_product(input_sections->image[(j * C1_DIMENSIONS) + k], network->C1_Kernels->kernels[0][i], KERNEL_SIZE) +
                    network->C1_Biases->vector[i]);

    free_image_vector(input_sections, KERNEL_SIZE, length);
    free(input_sections);
#endif

    // 计时累加
    clock_t _t_end = clock();
    conv1_total_time += (double)(_t_end - _t_start) / (double)CLOCKS_PER_SEC;
    conv1_calls++;
}


void load_S1(cnn *network)
{
    int i, n, j, k;
    image_vector *C1_sections = new_image_vector(2, S1_DIMENSIONS * S1_DIMENSIONS);
    // print_image(network->C1_Images->image[0], 24);
    for (i = 0; i < S1_LENGTH; i++)
    {
        for (n = 0; n < 2 * S1_DIMENSIONS * S1_DIMENSIONS; n++)
        {
            if (n % 2 == 0)
            {
                for (j = 0; j < 2; j++)
                {
                    for (k = 0; k < 2; k++)
                    {
                        C1_sections->image[n / 2]->matrix[j][k] =
                            network->C1_Images->image[i]->matrix[(int)floor((double)n / C1_DIMENSIONS) * 2 + j][(n % C1_DIMENSIONS) + k];
                    }
                }
                // if(i == 0)
                // print_image(C1_sections->image[n/2], 2);
            }
        }

        for (j = 0; j < S1_DIMENSIONS; j++)
        {
            for (k = 0; k < S1_DIMENSIONS; k++)
            {
                network->S1_Images->image[i]->matrix[j][k] = average_matrix(C1_sections->image[(j * 12) + k], 2);
            }
        }
    }
    // print_image(network->S1_Images->image[0], 12);
    free_image_vector(C1_sections, 2, S1_DIMENSIONS * S1_DIMENSIONS);
    free(C1_sections);
}

void load_C2(cnn *network)
{
    clock_t _t_start = clock(); // conv2 开始计时

#if USE_ISPC == 1
    // ISPC 并行分支
    int in_w = C1_DIMENSIONS;
    int in_h = C1_DIMENSIONS;
    int in_ch = S1_LENGTH;      // S1 输出通道数
    int out_ch = C2_LENGTH;
    int ksize = KERNEL_SIZE;
    int out_w = C2_DIMENSIONS;
    int out_h = C2_DIMENSIONS;

    // 分配并填充 input (多通道 S1)
    float *input_f = (float *)malloc(sizeof(float) * in_ch * in_w * in_h);
    for (int c = 0; c < in_ch; c++)
        for (int y = 0; y < in_h; y++)
            for (int x = 0; x < in_w; x++)
                input_f[(c * in_h + y) * in_w + x] = (float)network->S1_Images->image[c]->matrix[y][x];

    // 分配并填充 kernels: layout [out_ch][in_ch][ksize][ksize]
    float *kernels_f = (float *)malloc(sizeof(float) * out_ch * in_ch * ksize * ksize);
    for (int oc = 0; oc < out_ch; oc++)
        for (int ic = 0; ic < in_ch; ic++)
            for (int j = 0; j < ksize; j++)
                for (int k = 0; k < ksize; k++)
                    kernels_f[((oc * in_ch + ic) * ksize + j) * ksize + k] =
                        (float)network->C2_Kernels->kernels[ic][oc]->matrix[j][k];

    // biases -> float
    float *bias_f = (float *)malloc(sizeof(float) * out_ch);
    for (int oc = 0; oc < out_ch; oc++)
        bias_f[oc] = (float)network->C2_Biases->vector[oc];

    // 输出缓冲
    float *output_f = (float *)malloc(sizeof(float) * out_ch * out_w * out_h);

    // 包含 ISPC 头文件
    #include "conv_ispc.h"
    // 调用 ISPC 函数
    ispc_conv1(
        input_f,
        in_ch, in_w, in_h,
        kernels_f, ksize, in_ch, out_ch,
        bias_f,
        output_f,
        out_w, out_h
    );

    // 拷回 network->C2_Images 并应用激活
    for (int oc = 0; oc < out_ch; oc++)
        for (int y = 0; y < out_h; y++)
            for (int x = 0; x < out_w; x++)
            {
                double v = (double)output_f[(oc * out_h + y) * out_w + x];
                network->C2_Images->image[oc]->matrix[y][x] = activation(v);
            }

    free(input_f);
    free(kernels_f);
    free(bias_f);
    free(output_f);

#else
    // 串行分支保持原样
    image_vector *S1_sections[S1_LENGTH];
    int l, m, n, i, j, k;
    int length = C2_DIMENSIONS * C2_DIMENSIONS;
    for (i = 0; i < S1_LENGTH; i++)
        S1_sections[i] = new_image_vector(KERNEL_SIZE, length);

    for (n = 0; n < S1_LENGTH; n++)
        for (i = 0; i < length; i++)
            for (j = 0; j < KERNEL_SIZE; j++)
                for (k = 0; k < KERNEL_SIZE; k++)
                    S1_sections[n]->image[i]->matrix[j][k] =
                        network->S1_Images->image[n]->matrix[(int)floor((double)i / C2_DIMENSIONS) + j][i % C1_DIMENSIONS + k];

    for (m = 0; m < C2_LENGTH; m++)
        for (n = 0; n < C2_DIMENSIONS; n++)
            for (i = 0; i < C2_DIMENSIONS; i++)
                for (l = 0; l < S1_LENGTH; l++)
                    network->C2_Images->image[m]->matrix[n][i] +=
                        dot_product(S1_sections[l]->image[(n * C2_DIMENSIONS) + i], network->C2_Kernels->kernels[l][m], KERNEL_SIZE);

    for (i = 0; i < S1_LENGTH; i++)
    {
        free_image_vector(S1_sections[i], KERNEL_SIZE, length);
        free(S1_sections[i]);
    }
#endif

    // conv2 计时结束并累加
    clock_t _t_end = clock();
    conv2_total_time += (double)(_t_end - _t_start) / (double)CLOCKS_PER_SEC;
    conv2_calls++;
}


void load_S2(cnn *network)
{
    int i, n, j, k;
    image_vector *C2_sections = new_image_vector(2, S2_DIMENSIONS * S2_DIMENSIONS);
    // print_image(network->C2_Images->image[0],8);
    for (i = 0; i < C2_LENGTH; i++)
    {
        for (n = 0; n < 2 * S2_DIMENSIONS * S2_DIMENSIONS; n++)
        {
            if (n % 2 == 0)
            {
                for (j = 0; j < 2; j++)
                {
                    for (k = 0; k < 2; k++)
                    {
                        C2_sections->image[n / 2]->matrix[j][k] =
                            network->C2_Images->image[i]->matrix[(int)floor((double)n / C2_DIMENSIONS) * 2 + j][n % C2_DIMENSIONS + k];
                    }
                }
                // if(i == 0)
                // print_image(C2_sections->image[n/2], 2);
            }
        }

        for (j = 0; j < S2_DIMENSIONS; j++)
        {
            for (k = 0; k < S2_DIMENSIONS; k++)
            {
                network->S2_Images->image[i]->matrix[j][k] = average_matrix(C2_sections->image[(j * S2_DIMENSIONS) + k], 2);
            }
        }
        // print_image(network->S2_Images->image[i], 4);
    }
    free_image_vector(C2_sections, 2, S2_DIMENSIONS * S2_DIMENSIONS);
}

void concatenate_S2(cnn *network)
{
    // print_image(network->S2_Images->image[0],4);
    int n, j, k;
    for (n = 0; n < S2_LENGTH; n++)
    {
        for (j = 0; j < S2_DIMENSIONS; j++)
        {
            for (k = 0; k < S2_DIMENSIONS; k++)
            {
                network->S2_Vector->vector[(n * (S2_DIMENSIONS * S2_DIMENSIONS)) + (j * S2_DIMENSIONS) + k] = network->S2_Images->image[n]->matrix[j][k];
                // printf("%lf\n",network->S2_Vector->vector[(n*(S2_DIMENSIONS*S2_DIMENSIONS)) + (j*S2_DIMENSIONS) + k]);
            }
        }
    }
}

void load_output(cnn *network)
{
    int i, n, j, k;
    double value = 0;
    for (i = 0; i < OUTPUT_LENGTH; i++)
    {
        for (n = 0; n < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; n++)
        {
            value += (network->S2_Vector->vector[n] * network->output_weights->matrix[i][n]);
        }
        value += network->output_biases->vector[i];
        network->calculated_output->vector[i] = output_activation(value);
        value = 0;
    }
}

void backpropagation(cnn *network, cnn *gradient)
{
    int i, n, u, v, j, k, l, q;
    // Delta W
    for (i = 0; i < OUTPUT_LENGTH; i++)
    {
        for (j = 0; j < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; j++)
        {
            gradient->output_weights->matrix[i][j] +=
                (network->calculated_output->vector[i] - network->desired_output->vector[i]) * output_activation_derivative(network->calculated_output->vector[i]) * network->S2_Vector->vector[j];
        }
    }
    // Delta B
    for (i = 0; i < OUTPUT_LENGTH; i++)
    {
        gradient->output_biases->vector[i] +=
            (network->calculated_output->vector[i] - network->desired_output->vector[i]) * output_activation_derivative(network->calculated_output->vector[i]);
    }
    // Delta f(j)
    for (j = 0; j < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; j++)
    {
        for (i = 0; i < OUTPUT_LENGTH; i++)
        {
            gradient->S2_Vector->vector[j] +=
                (gradient->output_biases->vector[i] * network->output_weights->matrix[i][j]);
        }
    }
    // un-concatenate
    for (n = 0; n < S2_LENGTH; n++)
    {
        for (j = 0; j < S2_DIMENSIONS; j++)
        {
            for (k = 0; k < S2_DIMENSIONS; k++)
            {
                gradient->S2_Images->image[n]->matrix[j][k] =
                    gradient->S2_Vector->vector[(n * S2_DIMENSIONS * S2_DIMENSIONS) + (j * S2_DIMENSIONS) + k];
            }
        }
    }
    // super sample
    for (n = 0; n < C2_LENGTH; n++)
    {
        for (j = 0; j < C2_DIMENSIONS; j++)
        {
            for (k = 0; k < C2_DIMENSIONS; k++)
            {
                gradient->C2_Images->image[n]->matrix[j][k] =
                    (0.25) * gradient->S2_Images->image[n]->matrix[(int)floor((double)j / 2)][(int)floor((double)k / 2)];
            }
        }
    }
    // Delta K2
    for (n = 0; n < S1_LENGTH; n++)
    {
        for (i = 0; i < C2_LENGTH; i++)
        {
            for (u = 0; u < KERNEL_SIZE; u++)
            {
                for (v = 0; v < KERNEL_SIZE; v++)
                {
                    for (j = 0; j < C2_DIMENSIONS; j++)
                    {
                        for (k = 0; k < C2_DIMENSIONS; k++)
                        {
                            gradient->C2_Kernels->kernels[n][i]->matrix[u][v] +=
                                gradient->C2_Images->image[i]->matrix[j][k] *
                                activation_derivative(network->C2_Images->image[i]->matrix[j][k]) *
                                network->S1_Images->image[n]->matrix[j + u][k + v];
                        }
                    }
                }
            }
        }
    }
    // Delta B2
    for (n = 0; n < C2_LENGTH; n++)
    {
        for (j = 0; j < C2_DIMENSIONS; j++)
        {
            for (k = 0; k < C2_DIMENSIONS; k++)
            {
                gradient->C2_Biases->vector[n] +=
                    gradient->C2_Images->image[n]->matrix[j][k] *
                    activation_derivative(network->C2_Images->image[n]->matrix[j][k]);
            }
        }
    }
    // Funniest shit i've pulled so consider in debugging
    // Prep for backprop convolution
    image_vector *C2_gradient_temp = new_image_vector(S1_DIMENSIONS + KERNEL_SIZE - 1, C2_LENGTH);
    image_vector *C2_network_temp = new_image_vector(S1_DIMENSIONS + KERNEL_SIZE - 1, C2_LENGTH);
    for (i = 0; i < C2_LENGTH; i++)
    {
        for (j = 0; j < S1_DIMENSIONS + KERNEL_SIZE - 1; j++)
        {
            for (k = 0; k < S1_DIMENSIONS + KERNEL_SIZE - 1; k++)
            {
                C2_gradient_temp->image[i]->matrix[j][k] = 0;
                C2_network_temp->image[i]->matrix[j][k] = 0;
            }
        }
    }
    for (i = 0; i < C2_LENGTH; i++)
    {
        for (j = 4; j < C2_DIMENSIONS + 4; j++)
        {
            for (k = 4; k < C2_DIMENSIONS + 4; k++)
            {
                C2_gradient_temp->image[i]->matrix[j][k] = gradient->C2_Images->image[i]->matrix[j - 4][k - 4];
                C2_network_temp->image[i]->matrix[j][k] = network->C2_Images->image[i]->matrix[j - 4][k - 4];
            }
        }
    }
    // Delta Sp1
    for (n = 0; n < S1_LENGTH; n++)
    {
        for (i = 0; i < S1_DIMENSIONS; i++)
        {
            for (l = 0; l < S1_DIMENSIONS; l++)
            {
                for (q = 0; q < C2_LENGTH; q++)
                {
                    for (u = 0; u < KERNEL_SIZE; u++)
                    {
                        for (v = 0; v < KERNEL_SIZE; v++)
                        {
                            gradient->S1_Images->image[n]->matrix[i][l] +=
                                C2_gradient_temp->image[q]->matrix[i + u][l + v] *
                                activation_derivative(C2_network_temp->image[q]->matrix[i + u][l + v]) *
                                network->C2_Kernels->kernels[n][q]->matrix[u][v];
                        }
                    }
                }
            }
        }
    }
    free_image_vector(C2_gradient_temp, S1_DIMENSIONS + KERNEL_SIZE - 1, C2_LENGTH);
    free_image_vector(C2_network_temp, S1_DIMENSIONS + KERNEL_SIZE - 1, C2_LENGTH);
    free(C2_gradient_temp);
    free(C2_network_temp);

    // super sampling
    for (n = 0; n < C1_LENGTH; n++)
    {
        for (j = 0; j < C1_DIMENSIONS; j++)
        {
            for (k = 0; k < C1_DIMENSIONS; k++)
            {
                gradient->C1_Images->image[n]->matrix[j][k] =
                    (0.25) * gradient->S1_Images->image[n]->matrix[(int)floor((double)j / 2)][(int)floor((double)k / 2)];
            }
        }
    }
    // Delta K1
    for (n = 0; n < C1_LENGTH; n++)
    {
        for (u = 0; u < KERNEL_SIZE; u++)
        {
            for (v = 0; v < KERNEL_SIZE; v++)
            {
                for (j = 0; j < C1_DIMENSIONS; j++)
                {
                    for (k = 0; k < C1_DIMENSIONS; k++)
                    {
                        gradient->C1_Kernels->kernels[0][n]->matrix[u][v] +=
                            gradient->C1_Images->image[n]->matrix[j][k] *
                            activation_derivative(network->C1_Images->image[n]->matrix[j][k]) *
                            network->input_image->matrix[j + u][k + v];
                    }
                }
            }
        }
    }

    for (n = 0; n < C1_LENGTH; n++)
    {
        for (j = 0; j < C1_DIMENSIONS; j++)
        {
            for (k = 0; k < C1_DIMENSIONS; k++)
            {
                gradient->C1_Biases->vector[n] +=
                    gradient->C1_Images->image[n]->matrix[j][k];
            }
        }
    }
}

void update_batch_gradient(cnn *image_gradient, cnn *batch_gradient)
{
    int n, m, j, k;
    for (n = 0; n < C1_LENGTH; n++)
    {
        for (j = 0; j < KERNEL_SIZE; j++)
        {
            for (k = 0; k < KERNEL_SIZE; k++)
            {
                batch_gradient->C1_Kernels->kernels[0][n]->matrix[j][k] +=
                    image_gradient->C1_Kernels->kernels[0][n]->matrix[j][k];
            }
        }
        batch_gradient->C1_Biases->vector[n] += image_gradient->C1_Biases->vector[n];
    }
    for (n = 0; n < S1_LENGTH; n++)
    {
        for (m = 0; m < C2_LENGTH; m++)
        {
            for (j = 0; j < KERNEL_SIZE; j++)
            {
                for (k = 0; k < KERNEL_SIZE; k++)
                {
                    batch_gradient->C2_Kernels->kernels[n][m]->matrix[j][k] +=
                        image_gradient->C2_Kernels->kernels[n][m]->matrix[j][k];
                }
            }
            batch_gradient->C2_Biases->vector[m] +=
                image_gradient->C2_Biases->vector[m];
        }
    }
    for (n = 0; n < OUTPUT_LENGTH; n++)
    {
        for (m = 0; m < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; m++)
        {
            batch_gradient->output_weights->matrix[n][m] +=
                image_gradient->output_weights->matrix[n][m];
        }
        batch_gradient->output_biases->vector[n] +=
            image_gradient->output_biases->vector[n];
    }
}

void gradient_descent(cnn *network, cnn *gradient, int actual_batch_size)
{
    int n, m, j, k;
    for (n = 0; n < C1_LENGTH; n++)
    {
        for (j = 0; j < KERNEL_SIZE; j++)
        {
            for (k = 0; k < KERNEL_SIZE; k++)
            {
                network->C1_Kernels->kernels[0][n]->matrix[j][k] -=
                    LEARNING_RATE * (gradient->C1_Kernels->kernels[0][n]->matrix[j][k] / actual_batch_size);
            }
        }
        network->C1_Biases->vector[n] -= LEARNING_RATE * (gradient->C1_Biases->vector[n] / actual_batch_size);
    }
    for (n = 0; n < S1_LENGTH; n++)
    {
        for (m = 0; m < C2_LENGTH; m++)
        {
            for (j = 0; j < KERNEL_SIZE; j++)
            {
                for (k = 0; k < KERNEL_SIZE; k++)
                {
                    network->C2_Kernels->kernels[n][m]->matrix[j][k] -=
                        LEARNING_RATE * (gradient->C2_Kernels->kernels[n][m]->matrix[j][k] / actual_batch_size);
                }
            }
            network->C2_Biases->vector[m] -=
                LEARNING_RATE * (gradient->C2_Biases->vector[m] / actual_batch_size);
        }
    }
    for (n = 0; n < OUTPUT_LENGTH; n++)
    {
        for (m = 0; m < S2_LENGTH * S2_DIMENSIONS * S2_DIMENSIONS; m++)
        {
            network->output_weights->matrix[n][m] -=
                LEARNING_RATE * (gradient->output_weights->matrix[n][m] / actual_batch_size);
        }
        network->output_biases->vector[n] -=
            LEARNING_RATE * (gradient->output_biases->vector[n] / actual_batch_size);
    }
}

void free_image_vector(image_vector *images, int dimensions, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        free_array(images->image[i], dimensions);
        free(images->image[i]);
    }
    free(images->image);
}

void free_array(array *array, int rows)
{
    int j, k;
    for (j = 0; j < rows; j++)
    {
        free(array->matrix[j]);
    }
    free(array->matrix);
}

// Computes the average value of a matrix
double average_matrix(array *image, int dimensions)
{
    int j, k;
    double value = 0;
    for (j = 0; j < dimensions; j++)
    {
        for (k = 0; k < dimensions; k++)
        {
            value += image->matrix[j][k];
        }
    }
    value /= (dimensions * dimensions);
    return value;
}

void print_image(array *image, int dimensions)
{
    int j, k;
    for (j = 0; j < dimensions; j++)
    {
        for (k = 0; k < dimensions; k++)
        {
            if (image->matrix[j][k] > 0)
                printf(RED "[%.2lf]", image->matrix[j][k]);
            else
                printf(RESET "[%.2lf]", image->matrix[j][k]);
        }
        printf(RESET "\n");
    }
    printf(RESET "\n");
}

// Computes the dot product of two identically sized matrices
double dot_product(array *array1, array *array2, int dimensions)
{
    int j, k;
    double value = 0;
    for (j = 0; j < dimensions; j++)
    {
        for (k = 0; k < dimensions; k++)
        {
            value += array1->matrix[j][k] * array2->matrix[j][k];
        }
    }
    return value;
}

double activation(double x)
{
    if (x < 0)
        return 0.01 * x;
    return x;
}

double activation_derivative(double x)
{
    if (x < 0)
        return 0.01;
    return 1;
}

double output_activation(double x)
{
    return (double)1 / (1 + exp(-x));
}

double output_activation_derivative(double x)
{
    return x * (1 - x);
}

double loss_function(cnn *network)
{
    int i;
    double value = 0;
    for (i = 0; i < 10; i++)
    {
        value += (network->calculated_output->vector[i] - network->desired_output->vector[i]) * (network->calculated_output->vector[i] - network->desired_output->vector[i]);
    }
    return value / 2;
}
