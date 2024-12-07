#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <algorithm>
#include <chrono>

using namespace std;

// Функция для генерации ленточной матрицы
vector<vector<double>> generate_band_matrix(int rows, int cols, int band_width) {
    vector<vector<double>> matrix(rows, vector<double>(cols, 0));
    srand(time(0));
    for (int i = 0; i < rows; ++i) {
        // Заполнение элементов в пределах заданной ширины полосы
        for (int j = max(0, i - band_width); j <= min(cols - 1, i + band_width); ++j) {
            matrix[i][j] = rand() % 100 + 1;
        }
    }
    return matrix;
}

// Функция для генерации нижнетреугольной матрицы
vector<vector<double>> generate_lower_triangular_matrix(int rows, int cols) {
    vector<vector<double>> matrix(rows, vector<double>(cols, 0));
    srand(time(0));
    for (int i = 0; i < rows; ++i) {
        // Заполнение только нижней треугольной части матрицы
        for (int j = 0; j <= i; ++j) {
            matrix[i][j] = rand() % 100 + 1;
        }
    }
    return matrix;
}

// Функция для поиска минимального значения в строке
double find_min_in_row(const vector<double>& row) {
    double min_value = numeric_limits<double>::infinity();
    for (double val : row) {
        if (val != 0) min_value = min(min_value, val);
    }
    return min_value;
}

// Функция для поиска максимального значения среди минимальных в строках матрицы
double find_max_of_mins(const vector<vector<double>>& matrix, int num_threads, const string& schedule_type, int chunk_size, double& execution_time) {
    int num_rows = matrix.size();
    double max_min_value = -numeric_limits<double>::infinity();

    // Начало измерения времени
    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(num_threads);

    if (schedule_type == "static") {
        #pragma omp parallel for schedule(static, chunk_size) reduction(max:max_min_value)
        for (int i = 0; i < num_rows; ++i) {
            double min_in_row = find_min_in_row(matrix[i]);
            max_min_value = max(max_min_value, min_in_row);
        }
    } else if (schedule_type == "dynamic") {
        #pragma omp parallel for schedule(dynamic, chunk_size) reduction(max:max_min_value)
        for (int i = 0; i < num_rows; ++i) {
            double min_in_row = find_min_in_row(matrix[i]);
            max_min_value = max(max_min_value, min_in_row);
        }
    } else if (schedule_type == "guided") {
        #pragma omp parallel for schedule(guided, chunk_size) reduction(max:max_min_value)
        for (int i = 0; i < num_rows; ++i) {
            double min_in_row = find_min_in_row(matrix[i]);
            max_min_value = max(max_min_value, min_in_row);
        }
    }

    // Конец измерения времени
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    execution_time = duration.count();

    return max_min_value;
}

int main() {
    vector<int> matrix_sizes = {1000, 2000, 3000};
    vector<int> thread_counts = {2, 4, 8};
    int num_repeats = 3;
    int band_width = 5;
    int chunk_size = 10;
    vector<string> schedules = {"static", "dynamic", "guided"};

    cout << "Matrix Type   | Size   | Threads | Distribution  | Time (sec)  | Result\n";
    cout << "----------------------------------------------------------------------------\n";

    // Тесты для ленточной матрицы
    for (int size : matrix_sizes) {
        vector<vector<double>> band_matrix = generate_band_matrix(size, size, band_width);
        vector<vector<double>> lower_triangular_matrix = generate_lower_triangular_matrix(size, size);

        for (int threads : thread_counts) {
            for (const string& schedule_type : schedules) {
                double total_time = 0;
                double result = 0;

                for (int i = 0; i < num_repeats; ++i) {
                    double execution_time;
                    result = find_max_of_mins(band_matrix, threads, schedule_type, chunk_size, execution_time);
                    total_time += execution_time;
                }

                double avg_time = total_time / num_repeats;
                cout << setw(13) << "Band" << " | "
                     << setw(6) << size << " | "
                     << setw(7) << threads << " | "
                     << setw(12) << schedule_type << " | "
                     << setw(10) << avg_time << " | "
                     << setw(8) << result << "\n";
            }
        }

        // Тесты для нижнетреугольной матрицы
        for (int threads : thread_counts) {
            for (const string& schedule_type : schedules) {
                double total_time = 0;
                double result = 0;

                for (int i = 0; i < num_repeats; ++i) {
                    double execution_time;
                    result = find_max_of_mins(lower_triangular_matrix, threads, schedule_type, chunk_size, execution_time);
                    total_time += execution_time;
                }

                double avg_time = total_time / num_repeats;
                cout << setw(13) << "Triangular" << " | "
                     << setw(6) << size << " | "
                     << setw(7) << threads << " | "
                     << setw(12) << schedule_type << " | "
                     << setw(10) << avg_time << " | "
                     << setw(8) << result << "\n";
            }
        }
    }

    return 0;
}

