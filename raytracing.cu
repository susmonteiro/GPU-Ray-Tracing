#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define BLOCK_SIZE 8

#define HEADER_SIZE 138
#define FILE_HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define BYTES_PER_PIXEL 3 // red, green, & blue

typedef unsigned char BYTE;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__host__ __device__ float3 add_float3(float3 A, float3 B) {
    float3 C = {A.x + B.x, A.y + B.y, A.z + B.z};
    return C;
}

__host__ __device__ float3 sub_float3(float3 A, float3 B) {
    float3 C = {A.x - B.x, A.y - B.y, A.z - B.z};
    return C;
}

__host__ __device__ float dot_product(float3 v1, float3 v2) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

unsigned char* createBitmapFileHeader (int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] = {
        0,0,     /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
    };

    fileHeader[ 0] = (unsigned char)('B');
    fileHeader[ 1] = (unsigned char)('M');
    fileHeader[ 2] = (unsigned char)(fileSize      );
    fileHeader[ 3] = (unsigned char)(fileSize >>  8);
    fileHeader[ 4] = (unsigned char)(fileSize >> 16);
    fileHeader[ 5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

unsigned char* createBitmapInfoHeader (int height, int width)
{
    static unsigned char infoHeader[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0,     /// number of color planes
        0,0,     /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };

    infoHeader[ 0] = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[ 4] = (unsigned char)(width      );
    infoHeader[ 5] = (unsigned char)(width >>  8);
    infoHeader[ 6] = (unsigned char)(width >> 16);
    infoHeader[ 7] = (unsigned char)(width >> 24);
    infoHeader[ 8] = (unsigned char)(height      );
    infoHeader[ 9] = (unsigned char)(height >>  8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL*8);

    return infoHeader;
}


void saveImageCPU(int width, int height, float3** image, char filename[256]) {
    int widthInBytes = width * BYTES_PER_PIXEL;

    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes) + paddingSize;

    FILE* file = fopen(filename, "wb");

    unsigned char* fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, file);

    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, file);

    for (int w = 0; w < width; w++) {
        for (int h = height - 1; h > -1; h--) {
            double3 pixel = {image[h][w].x * 255., image[h][w].y * 255., image[h][w].z * 255.};
            char pixel_x = (char)((pixel.x > 255.0f) ? 255.0f :
                                (pixel.x < 0.0f)   ? 0.0f :
                                pixel.x);
            char pixel_y = (char)((pixel.y > 255.0f) ? 255.0f :
                                (pixel.y < 0.0f)   ? 0.0f :
                                pixel.y);
            char pixel_z = (char)((pixel.z > 255.0f) ? 255.0f :
                                (pixel.z < 0.0f)   ? 0.0f :
                                pixel.z);
            
            fputc(pixel_z, file); 
            fputc(pixel_y, file);
            fputc(pixel_x, file);
        }
    }

    fclose(file);
}

void saveImageGPU(int width, int height, float3* image, char filename[256]) {
    int widthInBytes = width * BYTES_PER_PIXEL;

    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes) + paddingSize;

    FILE* file = fopen(filename, "wb");

    unsigned char* fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, file);

    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, file);

    for (int w = 0; w < width; w++) {
        for (int h = height - 1; h > -1; h--) {
            double3 pixel = {image[h * height + w].x * 255., image[h * height + w].y * 255., image[h * height + w].z * 255.};
            char pixel_x = (char)((pixel.x > 255.0f) ? 255.0f :
                                (pixel.x < 0.0f)   ? 0.0f :
                                pixel.x);
            char pixel_y = (char)((pixel.y > 255.0f) ? 255.0f :
                                (pixel.y < 0.0f)   ? 0.0f :
                                pixel.y);
            char pixel_z = (char)((pixel.z > 255.0f) ? 255.0f :
                                (pixel.z < 0.0f)   ? 0.0f :
                                pixel.z);
            
            fputc(pixel_z, file); 
            fputc(pixel_y, file);
            fputc(pixel_x, file);
        }
    }

    fclose(file);
}

__host__ __device__ float3 normalize(float3 v) {
    float norma = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= norma;
    v.y /= norma;
    v.z /= norma;
    return v;
}

/* Return the distance from O to the intersection of the ray (O, D) with the sphere (S, R) 
    O and S are 3D points, D (direction) is a normalised vector and R is a scalar */
__host__ __device__ float intersect_sphere(float3 O, float3 D, float3 S, float R) {
    float a = dot_product(D, D);
    float3 OS = sub_float3(O, S);
    float b = 2 * dot_product(D, OS);
    float c = dot_product(OS, OS) - R * R;
    float disc = b*b - 4*a*c;
    if (disc > 0) {
        float discSqrt = sqrt(disc);
        float q = b < 0 ? (-b - discSqrt) / 2.0 : (-b + discSqrt) / 2.0;
        float t0 = q / a;
        float t1 = c / q;
        float tmin = min(t0, t1);
        float tmax = max(t0, t1);
        if (t1 >= 0) return t0 < 0 ? t1 : t0;
    }
    return INFINITY;
}

/* Find first point of intersection with the scene, trace a ray and apply Bling-phon shading */
__host__ __device__ float3 trace_ray(float3 O, float3 D, float3 position, float radius, float3 L, float ambient, 
            float diffuse, float3 color, float specular_c, int specular_k, float3 color_light) {
    float t = intersect_sphere(O, D, position, radius);
    if (t == INFINITY) return (float3){INFINITY, INFINITY, INFINITY}; // means no intersection

    float3 M = {O.x + D.x*t, O.y + D.y*t, O.z + D.z*t};

    float3 N = normalize(sub_float3(M, position));
    float3 toL = normalize(sub_float3(L, M));
    float3 toO = normalize(sub_float3(O, M));

    float3 col;
    float d_prod = max(dot_product(N, toL), 0.);
    col.x = ambient + diffuse * d_prod * color.x;
    col.y = ambient + diffuse * d_prod * color.y;
    col.z = ambient + diffuse * d_prod * color.z;

    d_prod = max(dot_product(N, normalize(add_float3(toL, toO))), 0.);
    col.x += specular_c * color_light.x * pow(d_prod, specular_k);
    col.y += specular_c * color_light.y * pow(d_prod, specular_k);
    col.z += specular_c * color_light.z * pow(d_prod, specular_k);

    return col;
}

float3** cpu_compute(int width, int height, float3 O, float3 Q, float3 position, float radius, float3 L, 
            float ambient, float diffuse, float3 color, float specular_c, int specular_k, float3 color_light, double step) {
    float3 **img = (float3**)malloc(height * sizeof(float3*));
    for (int h = 0; h < height; h++) img[h] = (float3*)calloc(width, sizeof(float3));

    int counterWidth = 0;
    int counterHeight = 0;
    for (double w = -1.; w < 1.005; w += step) {
        counterHeight = -1;
        for (double h = -1.; h < 1.005; h += step) {
            counterHeight++;
            Q.x = w, Q.y = h;

            float3 D = normalize(sub_float3(Q, O));
            float3 col = trace_ray(O, D, position, radius, L, ambient, diffuse, color, specular_c, specular_k, color_light);
            if (col.x == INFINITY) continue;

            col.x = col.x > 1 ? 1 : col.x < 0 ? 0 : col.x; 
            col.y = col.y > 1 ? 1 : col.y < 0 ? 0 : col.y; 
            col.z = col.z > 1 ? 1 : col.z < 0 ? 0 : col.z; 

            img[height - counterHeight - 1][counterWidth] = col;
        }
        counterWidth++;
    }
    return img;
}

__global__ void gpu_compute(int width, int height, float3 O, float3 Q, float3 position, float radius, float3 L, 
            float ambient, float diffuse, float3 color, float specular_c, int specular_k, float3 color_light, 
            double step, float3 *img) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (!(x >= 0 && x < width && y >= 0 && y < height)) return;

    Q.x = -1 + x*step, Q.y = -1+y*step;
    float3 D = normalize(sub_float3(Q, O));
    float3 col = trace_ray(O, D, position, radius, L, ambient, diffuse, color, specular_c, specular_k, color_light);

    if (col.x == INFINITY) { 
        col.x = 0, col.y = 0, col.z = 0;
    } else {
        col.x = col.x > 1 ? 1 : col.x < 0 ? 0 : col.x; 
        col.y = col.y > 1 ? 1 : col.y < 0 ? 0 : col.y; 
        col.z = col.z > 1 ? 1 : col.z < 0 ? 0 : col.z; 

    }
    img[height*(height - x - 1) + y] = col;
}

int main() {
    printf("Block Size: %d\n", BLOCK_SIZE*BLOCK_SIZE);
    int width = 4000, height = 4000;

    // sphere properties
    float3 position = {0., 0., 1.};
    float radius = 1.;
    float3 color = {0., 0., 1.};
    float diffuse = 1.;
    float specular_c = 1.;
    int specular_k = 50;

    // light position and color
    float3 L = {5., 5., -10.};
    float3 color_light = {1., 1., 1.};
    float ambient = 0.05;

    // camera
    float3 O = {0., 0., -1.};
    float3 Q = {0., 0., 0.};

    double step = 2. / (width - 1);

    // compute in cpu
    double iStart = cpuSecond();

    printf("Computing on CPU...\n");
    float3** img_cpu = cpu_compute(width, height, O, Q, position, radius, L, ambient, diffuse, color, specular_c, specular_k, color_light, step);
    
    double iCPUElaps = cpuSecond() - iStart;
    printf("Time elapsed CPU: %f\n\n", iCPUElaps);

    // compute in gpu
    double iStartMem = cpuSecond();

    dim3 grid(((width  + (BLOCK_SIZE - 1)) / BLOCK_SIZE),
                      ((height + (BLOCK_SIZE - 1)) / BLOCK_SIZE));                       
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    float3 *d_img_gpu;
    cudaMalloc((void**)&d_img_gpu, height * width * sizeof(float3));
    float3* img_gpu = (float3*)malloc(height*width*sizeof(float3));

    iStart = cpuSecond();
    printf("Computing on GPU...\n");
    gpu_compute<<<grid, block>>>(width, height, O, Q, position, radius, L, ambient, diffuse, color, specular_c, specular_k, color_light, step, d_img_gpu);
    cudaDeviceSynchronize();
    double iGPUElaps = cpuSecond() - iStart;

    cudaMemcpy(img_gpu, d_img_gpu, sizeof(float3)*width*height, cudaMemcpyDeviceToHost);

    double iGPUMemElaps = cpuSecond() - iStartMem;

    printf("Time elapsed GPU (not counting data transfer time): %f\n", iGPUElaps);
    printf("Time elapsed GPU (counting data transfer time):     %f\n\n", iGPUMemElaps);


    int cnt = 0;
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            if (abs(img_cpu[h][w].x - img_gpu[h * height + w].x) < 0.0001 &&
                abs(img_cpu[h][w].y - img_gpu[h * height + w].y) < 0.0001 &&
                abs(img_cpu[h][w].z - img_gpu[h * height + w].z) < 0.0001)
                    ++cnt;
        }
    }


    if (cnt == height*width) printf("Comparing the output for each implementation... Correct!\n\n"); 
    else printf("Comparing the output for each implementation... Incorrect :(\n\n");


    printf("Printing image...\n");

    char cpu_filename[256] = "images/raytracing_cpu.bmp";
    char gpu_filename[256] = "images/raytracing_gpu.bmp";

    saveImageCPU(width, height, img_cpu, cpu_filename);
    saveImageGPU(width, height, img_gpu, gpu_filename);

    printf("Done!\n");

    cudaFree(d_img_gpu);

    for(int i=0; i<height; i++) free(img_cpu[i]); 
    free(img_cpu);
    free(img_gpu);


    return 0;
}