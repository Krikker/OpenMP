#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <chrono>

using namespace std;

// Функция для вычисления скалярного произведения двух векторов
void compute_dot_product(int vector_size, int num_threads, int num_repeats) {
    vector<double> A(vector_size, 1.0);
    vector<double> B(vector_size, 2.0);
    double dot_product = 0.0;

    // Установка количества потоков
    omp_set_num_threads(num_threads);

    double total_time = 0.0;

    for (int r = 0; r < num_repeats; ++r) {
        dot_product = 0.0;

        // Начало измерения времени
        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(+:dot_product)
        for (int i = 0; i < vector_size; ++i) {
            dot_product += A[i] * B[i];
        }

        // Конец измерения времени
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        total_time += duration.count();
    }

    double average_time = total_time / num_repeats;

    cout << setw(15) << vector_size << " | "
         << setw(10) << num_threads << " | "
         << setw(15) << average_time << " s | "
         << setw(20) << dot_product << endl;
}

int main() {
    cout << "Vector Size    | Threads   | Execution Time  | Dot Product\n";
    cout << "------------------------------------------------------------------\n";

    vector<int> vector_sizes = {10000, 100000, 1000000, 100000000};
    vector<int> thread_counts = {1, 2, 4, 8, 12, 16};
    int num_repeats = 10;

    // Запускаем тесты для всех размеров векторов и всех вариантов числа потоков
    for (int size : vector_sizes) {
        for (int threads : thread_counts) {
            compute_dot_product(size, threads, num_repeats);
        }
    }

    return 0;
}