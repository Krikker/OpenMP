#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <algorithm>
#include <chrono>

using namespace std;

// Функция для поиска минимального элемента в строке матрицы
double find_min_in_row(const vector<double>& row) {
    return *min_element(row.begin(), row.end());
}

// Функция для нахождения максимального значения среди минимальных элементов строк матрицы
double find_max_of_mins(const vector<vector<double>>& matrix, int num_threads, int num_repeats, double& avg_time) {
    int num_rows = matrix.size();
    double max_min_value = -numeric_limits<double>::infinity(); // Начальное значение для максимума
    omp_set_num_threads(num_threads);
    double total_time = 0.0;
    for (int r = 0; r < num_repeats; ++r) {
        max_min_value = -numeric_limits<double>::infinity();  // Сбрасываем максимум перед каждой итерацией
        
        // Начало измерения времени
        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(max:max_min_value)
        for (int i = 0; i < num_rows; ++i) {
            double min_in_row = find_min_in_row(matrix[i]);
            max_min_value = max(max_min_value, min_in_row);
        }

        // Конец измерения времени
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        total_time += duration.count();
    }

    avg_time = total_time / num_repeats;
    return max_min_value;
}

int main() {
    cout << "Number of Rows     | Threads    | Execution Time  | Max of Row Minimums\n";
    cout << "---------------------------------------------------------------\n";
    vector<int> row_counts = {1000, 5000, 10000};
    vector<int> thread_counts = {1, 2, 4, 8, 16};
    int num_cols = 100;
    int num_repeats = 10;

    // Основной цикл по размерам матриц
    for (int rows : row_counts) {
        // Инициализация матрицы случайными числами
        vector<vector<double>> matrix(rows, vector<double>(num_cols));
        srand(time(0));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                matrix[i][j] = rand() % 100 + 1;  // Заполняем матрицу числами от 1 до 100
            }
        }

        // Цикл по количеству потоков
        for (int threads : thread_counts) {
            double avg_time;
            double result = find_max_of_mins(matrix, threads, num_repeats, avg_time);
            cout << setw(18) << rows << " | "
                 << setw(10) << threads << " | "
                 << setw(15) << avg_time << " s | "
                 << setw(15) << result << endl;
        }
    }

    return 0;
}
