#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

using namespace std;

// Функция для инициализации матрицы случайными значениями
void initialize_matrix(vector<vector<int>>& matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[i][j] = rand() % 100;
        }
    }
}

// Функция для поиска максимального значения среди минимальных элементов строк (без вложенного параллелизма)
double find_max_of_mins_no_nested_parallel(const vector<vector<int>>& matrix, int num_threads) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    int max_of_mins = matrix[0][0];
    
    omp_set_num_threads(num_threads);
    auto start = chrono::high_resolution_clock::now();

    // Параллельный цикл по строкам матрицы
    #pragma omp parallel for reduction(max:max_of_mins)
    for (size_t i = 0; i < rows; ++i) {
        int min_in_row = matrix[i][0];
        for (size_t j = 1; j < cols; ++j) {
            if (matrix[i][j] < min_in_row) {
                min_in_row = matrix[i][j];
            }
        }
        max_of_mins = max(max_of_mins, min_in_row);
    }

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

// Функция для поиска максимального значения среди минимальных элементов строк (с вложенным параллелизмом)
double find_max_of_mins_with_nested_parallel(const vector<vector<int>>& matrix, int num_threads) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    int max_of_mins = matrix[0][0];
    
    omp_set_num_threads(num_threads);
    auto start = chrono::high_resolution_clock::now();

    // Внешний параллельный цикл по строкам матрицы
    #pragma omp parallel for shared(matrix) reduction(max:max_of_mins)
    for (size_t i = 0; i < rows; ++i) {
        int min_in_row = matrix[i][0];

        // Вложенный параллельный цикл по элементам строки
        #pragma omp parallel for reduction(min:min_in_row)
        for (size_t j = 1; j < cols; ++j) {
            min_in_row = min(min_in_row, matrix[i][j]);
        }
        
        max_of_mins = max(max_of_mins, min_in_row);
    }

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end - start).count();
}

int main() {
    const vector<int> thread_counts = {2, 4, 8, 16};
    const vector<int> matrix_sizes = {100, 500, 1000};
    
    // Включаем вложенный параллелизм
    omp_set_nested(1);

    // Проверяем, включен ли вложенный параллелизм
    if (omp_get_nested()) {
        cout << "Nested parallelism is enabled." << endl;
    } else {
        cout << "Nested parallelism is not enabled." << endl;
    }

    cout << "Method | Number of Threads | Matrix Size | Time (sec)\n";
    cout << "--------------------------------------------------------------\n";

    // Основной цикл по размерам матрицы
    for (int size : matrix_sizes) {
        vector<vector<int>> matrix(size, vector<int>(size));
        initialize_matrix(matrix, size, size);
        for (int num_threads : thread_counts) {
            double time_no_nested = find_max_of_mins_no_nested_parallel(matrix, num_threads);
            double time_with_nested = find_max_of_mins_with_nested_parallel(matrix, num_threads);

            cout << fixed << setprecision(6);
            cout << "Without Nested Parallelism | " << num_threads << "              | " << size << "x" << size << "         | " << time_no_nested << "\n";
            cout << "With Nested Parallelism    | " << num_threads << "              | " << size << "x" << size << "         | " << time_with_nested << "\n";
            cout << "--------------------------------------------------------------\n";
        }
    }

    return 0;
}
