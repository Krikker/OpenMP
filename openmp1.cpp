#include <iostream>
#include <vector>
#include <chrono>
#include <limits>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

using namespace std;

// Функция для поиска минимума и максимума с использованием редукции
void find_min_max_reduction(const vector<int>& vec, int num_threads, int num_repeats) {
    int min_val = numeric_limits<int>::max();
    int max_val = numeric_limits<int>::lowest();

    // Установка количества потоков
    omp_set_num_threads(num_threads);

    double total_time = 0.0;

    for (int r = 0; r < num_repeats; ++r) {
        min_val = numeric_limits<int>::max();
        max_val = numeric_limits<int>::lowest();

        // Начало измерения времени
        auto start = chrono::high_resolution_clock::now();

        // Параллельный цикл с reduction для обновления min_val и max_val
        #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] < min_val) min_val = vec[i];
            if (vec[i] > max_val) max_val = vec[i];
        }

        // Конец измерения времени
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        total_time += duration.count();
    }

    double average_time = total_time / num_repeats;

    cout << setw(10) << vec.size() << " | "
         << setw(10) << num_threads << " | "
         << setw(15) << average_time << " s | "
         << "Min: " << setw(10) << min_val << ", Max: " << setw(10) << max_val
         << " (with reduction)" << endl;
}

// Функция для поиска минимума и максимума без использования редукции
void find_min_max_no_reduction(const vector<int>& vec, int num_threads, int num_repeats) {
    int min_val = numeric_limits<int>::max();
    int max_val = numeric_limits<int>::lowest();

    omp_set_num_threads(num_threads);

    double total_time = 0.0;

    for (int r = 0; r < num_repeats; ++r) {
        min_val = numeric_limits<int>::max();
        max_val = numeric_limits<int>::lowest();

        // Начало измерения времени
        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel
        {
            int local_min = numeric_limits<int>::max();
            int local_max = numeric_limits<int>::lowest();

            // Параллельный цикл для обновления локальных значений
            #pragma omp for
            for (size_t i = 0; i < vec.size(); ++i) {
                if (vec[i] < local_min) local_min = vec[i];
                if (vec[i] > local_max) local_max = vec[i];
            }

            // Критическая секция для обновления глобальных значений
            #pragma omp critical
            {
                if (local_min < min_val) min_val = local_min;
                if (local_max > max_val) max_val = local_max;
            }
        }

        // Конец измерения времени
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        total_time += duration.count();
    }

    double average_time = total_time / num_repeats;

    cout << setw(10) << vec.size() << " | "
         << setw(10) << num_threads << " | "
         << setw(15) << average_time << " s | "
         << "Min: " << setw(10) << min_val << ", Max: " << setw(10) << max_val
         << " (without reduction)" << endl;
}

int main() {
    cout << "Vector Size   | Threads   | Execution Time  | Min and Max Values\n";
    cout << "-------------------------------------------------------------------\n";

    vector<int> vector_sizes = {10000, 100000, 1000000, 10000000};
    vector<int> thread_counts = {1, 2, 4, 8, 16};
    int num_repeats = 10;

    // Основной цикл по размерам векторов
    for (int size : vector_sizes) {
        // Генерация случайного вектора
        vector<int> vec(size);
        for (int i = 0; i < size; ++i) {
            vec[i] = rand() % 10000 + 1;
        }

        // Цикл по количеству потоков
        for (int threads : thread_counts) {
            find_min_max_reduction(vec, threads, num_repeats);
            find_min_max_no_reduction(vec, threads, num_repeats);
        }
    }

    return 0;
}
