#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <chrono>

using namespace std;

// Функция, которую нужно интегрировать
double function(double x) {
    return x * x * x;
}

// Функция для вычисления интеграла методом средних прямоугольников
double compute_integral(double a, double b, int n, int num_threads, int num_repeats, double& avg_time) {
    double h = (b - a) / n;  // Шаг разбиения

    // Установка количества потоков
    omp_set_num_threads(num_threads);
    double total_time = 0.0;
    double integral = 0.0;
    for (int r = 0; r < num_repeats; ++r) {
        integral = 0.0;
        
        // Начало измерения времени
        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(+:integral)
        for (int i = 0; i < n; ++i) {
            double x = a + (i + 0.5) * h;  // Центр i-го отрезка
            integral += function(x);  // Суммируем значения функции в точках
        }

        // Конец измерения времени
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        total_time += duration.count();
    }

    avg_time = total_time / num_repeats;

    // Умножаем на шаг h, чтобы получить окончательное значение интеграла
    integral *= (b - a) / n;
    return integral;
}

int main() {
    cout << "Number of Divisions | Threads | Execution Time | Integral Value\n";
    cout << "---------------------------------------------------------------\n";

    double a = 0.0, b = 1.0;  // Границы интегрирования
    vector<int> divisions = {10000, 100000, 1000000};
    vector<int> thread_counts = {1, 2, 4, 8, 16};
    int num_repeats = 10;

    // Внешний цикл по количеству разбиений
    for (int n : divisions) {
        for (int threads : thread_counts) {
            double avg_time;
            double integral = compute_integral(a, b, n, threads, num_repeats, avg_time);
            cout << setw(18) << n << " | "
                 << setw(10) << threads << " | "
                 << setw(15) << avg_time << " s | "
                 << setw(15) << integral << endl;
        }
    }

    return 0;
}
