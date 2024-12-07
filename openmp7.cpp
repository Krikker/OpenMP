#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;

// Инициализация вектора случайными значениями от 0 до 99
void initialize_vector(vector<int>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = rand() % 100;
    }
}

// Суммирование элементов с использованием атомарной операции
double reduction_atomic(const vector<int>& vec, int num_threads) {
    int sum = 0;
    omp_set_num_threads(num_threads);
    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); ++i) {
        #pragma omp atomic  // Атомарное сложение
        sum += vec[i];
    }

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

// Суммирование элементов с использованием критической секции
double reduction_critical(const vector<int>& vec, int num_threads) {
    int sum = 0;
    omp_set_num_threads(num_threads);
    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); ++i) {
        #pragma omp critical  // Синхронизация потоков через критическую секцию
        sum += vec[i];
    }

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

// Суммирование элементов с использованием замков
double reduction_lock(const vector<int>& vec, int num_threads) {
    int sum = 0;
    omp_lock_t lock;  // Инициализация замка
    omp_init_lock(&lock);
    omp_set_num_threads(num_threads);
    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); ++i) {
        omp_set_lock(&lock);  // Захват замка
        sum += vec[i];
        omp_unset_lock(&lock);  // Освобождение замка
    }

    omp_destroy_lock(&lock);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

// Суммирование элементов с использованием встроенной конструкции редукции
double reduction_builtin(const vector<int>& vec, int num_threads) {
    int sum = 0;
    omp_set_num_threads(num_threads);
    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

int main() {
    const vector<int> thread_counts = {2, 4, 8, 16};
    const vector<int> vector_sizes = {10000, 100000, 1000000};

    std::cout << "Method | Number of Threads | Vector Size | Time (seconds)\n";
    cout << "--------------------------------------------------------------\n";

    // Основной цикл по размерам вектора
    for (int vector_size : vector_sizes) {
        vector<int> vec(vector_size);
        initialize_vector(vec);

        for (int num_threads : thread_counts) {
            double time_atomic = reduction_atomic(vec, num_threads);
            double time_critical = reduction_critical(vec, num_threads);
            double time_lock = reduction_lock(vec, num_threads);
            double time_builtin = reduction_builtin(vec, num_threads);

            cout << fixed << setprecision(6);
            cout << "Atomic Operation      | " << num_threads << "           | " << vector_size << "       | " << time_atomic << "\n";
            cout << "Critical Section      | " << num_threads << "           | " << vector_size << "       | " << time_critical << "\n";
            cout << "Lock                  | " << num_threads << "           | " << vector_size << "       | " << time_lock << "\n";
            cout << "Built-in Reduction    | " << num_threads << "           | " << vector_size << "       | " << time_builtin << "\n";
            cout << "--------------------------------------------------------------\n";
        }
    }

    return 0;
}
