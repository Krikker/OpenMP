#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iomanip>

using namespace std;

// Имитация ресурсоемкой вычислительной задачи
void heavy_computation(int i) {
    int iterations = rand() % 1000 + 1;
    double sum = 0;
    for (int j = 0; j < iterations; ++j) {
        sum += j * 0.0001;
    }
}

// Функция для тестирования различных типов распределения итераций
void test_schedule(int num_threads, int num_iterations, const string& schedule_type, int chunk_size) {

    omp_set_num_threads(num_threads);

    // Начало измерения времени
    auto start = chrono::high_resolution_clock::now();

    // Выполняем цикл с различными типами распределения итераций
    if (schedule_type == "static") {
        #pragma omp parallel for schedule(static, chunk_size)
        for (int i = 0; i < num_iterations; ++i) {
            heavy_computation(i);
        }
    } else if (schedule_type == "dynamic") {
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < num_iterations; ++i) {
            heavy_computation(i);
        }
    } else if (schedule_type == "guided") {
        #pragma omp parallel for schedule(guided, 100)
        for (int i = 0; i < num_iterations; ++i) {
            heavy_computation(i);
        }
    }

    // Конец измерения времени
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    cout << "Mode: " << setw(7) << schedule_type
         << " | Number of threads: " << setw(2) << num_threads
         << " | Execution time: " << setw(10) << duration.count() << " sec\n";
}

int main() {
    int num_iterations = 10000;
    int chunk_size = 10;
    vector<int> thread_counts = {2, 4, 8};

    cout << "Experimenting with iteration scheduling modes:\n";
    cout << "---------------------------------------------------\n";

    // Перебираем разные варианты числа потоков и распределения итераций
    for (int num_threads : thread_counts) {
        cout << "Number of threads: " << num_threads << "\n";
        test_schedule(num_threads, num_iterations, "static", chunk_size);
        test_schedule(num_threads, num_iterations, "dynamic", chunk_size);
        test_schedule(num_threads, num_iterations, "guided", chunk_size);
        cout << "---------------------------------------------------\n";
    }

    return 0;
}
