#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define HEADER_SIZE 138

typedef unsigned char BYTE;

float3 add_float3(float3 A, float3 B) {
    float3 C = {A.x + B.x, A.y + B.y, A.z + B.z};
    return C;
}

float3 sub_float3(float3 A, float3 B) {
    float3 C = {A.x - B.x, A.y - B.y, A.z - B.z};
    return C;
}

float dot_product(float3 v1, float3 v2) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}


void saveImage(int width, int height, float3** image, bool gpu) {
    char path[255] = "images/raytracing_cpu.bmp";
    FILE *file = NULL; 
    file = fopen(path, "wb");

    // fwrite(g_info, sizeof(BYTE), HEADER_SIZE, file); TODO uncomment me

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            BYTE pixel_x = (BYTE)((image[h][w].x > 255.0f) ? 255.0f :
                                (image[h][w].x < 0.0f)   ? 0.0f :
                                image[h][w].x);
            BYTE pixel_y = (BYTE)((image[h][w].y > 255.0f) ? 255.0f :
                                (image[h][w].y < 0.0f)   ? 0.0f :
                                image[h][w].y);
            BYTE pixel_z = (BYTE)((image[h][w].z > 255.0f) ? 255.0f :
                                (image[h][w].z < 0.0f)   ? 0.0f :
                                image[h][w].z);

        fputc(pixel_x, file); // TODO change me
        fputc(pixel_y, file); // TODO change me
        fputc(pixel_z, file); // TODO change me
        }
    }

    fclose(file);
}

float3 normalize(float3 v) {
    float norma = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= norma;
    v.y /= norma;
    v.z /= norma;
    return v;
}

/* Return the distance from O to the intersection of the ray (O, D) with the sphere (S, R) 
    O and S are 3D points, D (direction) is a normalised vector and R is a scalar */
float intersect_sphere(float3 O, float3 D, float3 S, float R) {
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
float3 trace_ray(float3 O, float3 D, float3 position, float radius, float3 L, float ambient, 
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
            float ambient, float diffuse, float3 color, float specular_c, int specular_k, float3 color_light) {
    float3 **img = (float3**)malloc(height * sizeof(float3*));
    for (int h = 0; h < height; h++) img[h] = (float3*)calloc(width, sizeof(float3));

    double step = 2. / (width - 1);
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

int main() {
    int width = 5, height = 5;

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

    float3** img = cpu_compute(width, height, O, Q, position, radius, L, ambient, diffuse, color, specular_c, specular_k, color_light);

    printf("Printing image...\n");

    saveImage(width, height, img, false);
    printf("Done!\n");
    return 0;
}