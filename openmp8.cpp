#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <mutex>
#include <condition_variable>

using namespace std;

vector<int> buffer1_1, buffer2_1;
vector<int> buffer1_2, buffer2_2;
bool useFirstBuffer = true;
mutex mtx;  // Мьютекс для синхронизации потоков
condition_variable cv;
bool done = false;

// Генерация случайных векторов и запись их в файл
void generateAndWriteVectors(const string& filename, int n, int dim) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file for writing!" << endl;
        return;
    }

    srand(time(0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) {
            int element = rand() % 10;
            file << element << " ";
        }
        file << endl;
    }

    file.close();
}

// Чтение пар векторов из файла и сохранение в буферы
void readVectorsPairwise(const string& filename, int dim) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file!" << endl;
        done = true;
        return;
    }

    int element;
    int vectorCount = 0;
    vector<int> currentVector;
    currentVector.reserve(dim);

    while (file >> element) {
        currentVector.push_back(element);

        // Если вектор заполнен, добавляем его в буфер
        if (currentVector.size() == dim) {
            vectorCount++;
            unique_lock<mutex> lock(mtx);

            if (vectorCount % 2 == 1) {
                if (useFirstBuffer) {
                    buffer1_1 = currentVector;
                } else {
                    buffer1_2 = currentVector;
                }
            } else {
                if (useFirstBuffer) {
                    buffer2_1 = currentVector;
                } else {
                    buffer2_2 = currentVector;
                }
                
                useFirstBuffer = !useFirstBuffer; // Переключаем буфер
                cv.notify_one(); // Уведомляем другой поток
                cv.wait(lock);   // Ждём завершения обработки
            }
            currentVector.clear();
        }
    }

    done = true;
    cv.notify_one();  // Уведомляем другой поток о завершении
    file.close();
}

// Вычисление скалярного произведения пар векторов
void calculateDotProduct(int dim, vector<int>& results) {
    while (true) {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [] { return (done || (!buffer1_1.empty() && !buffer2_1.empty()) || (!buffer1_2.empty() && !buffer2_2.empty())); });

        // Проверяем, если завершено чтение и буферы пусты
        if (done && (buffer1_1.empty() || buffer2_1.empty()) && (buffer1_2.empty() || buffer2_2.empty())) {
            break;
        }

        int partialSum = 0;
        if (useFirstBuffer) {
            for (int i = 0; i < dim; ++i) {
                partialSum += buffer1_2[i] * buffer2_2[i];
            }
            results.push_back(partialSum);  // Добавляем результат
            buffer1_2.clear();
            buffer2_2.clear();
        } else {
            for (int i = 0; i < dim; ++i) {
                partialSum += buffer1_1[i] * buffer2_1[i];
            }
            results.push_back(partialSum);  // Добавляем результат
            buffer1_1.clear();
            buffer2_1.clear();
        }

        cv.notify_one();  // Уведомляем другой поток
    }
}

// Последовательное вычисление скалярного произведения для проверки
void calculateDotProductSequential(const string& filename, int dim, int n, vector<int>& results) {
    ifstream file(filename);
    vector<int> vec1(dim), vec2(dim);
    int element;
    
    for (int i = 0; i < n / 2; ++i) {
        for (int j = 0; j < dim; ++j) {
            file >> vec1[j];
        }
        
        for (int j = 0; j < dim; ++j) {
            file >> vec2[j];
        }
        
        int dotProduct = 0;
        for (int j = 0; j < dim; ++j) {
            dotProduct += vec1[j] * vec2[j];
        }
        results.push_back(dotProduct);
    }
    file.close();
}

int main() {
    string filename = "vectors.txt";
    vector<int> vector_counts = {1000, 2000, 3000};
    vector<int> matrix_sizes = {1000, 2000, 3000};
    vector<int> thread_counts = {2,4,8};

    cout << "Number of vectors | Vector size | Threads  | Time (sec) | Result\n";

    for (int n : vector_counts) {
        for (int dim : matrix_sizes) {
            for (int threads : thread_counts) {
                done = false;  // Сбрасываем флаг завершения
                generateAndWriteVectors(filename, n, dim);  // Генерируем данные

                vector<int> parallelResults;
                vector<int> sequentialResults;

                // Последовательное вычисление
                auto startSeq = chrono::high_resolution_clock::now();
                calculateDotProductSequential(filename, dim, n, sequentialResults);
                auto endSeq = chrono::high_resolution_clock::now();
                auto sequentialTime = chrono::duration<double>(endSeq - startSeq).count();
                omp_set_num_threads(threads);
                auto startPar = chrono::high_resolution_clock::now();
                // Параллельное выполнение
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        readVectorsPairwise(filename, dim);
                    }

                    #pragma omp section
                    {
                        calculateDotProduct(dim, parallelResults);
                    }
                }
                auto endPar = chrono::high_resolution_clock::now();
                auto parallelTime = chrono::duration<double>(endPar - startPar).count();

                cout << n << " | " << dim << " | " << threads << " | ";
                cout << parallelTime << " | ";
                if (parallelResults == sequentialResults) {
                    cout << "Match\n";
                } else {
                    cout << "Do not match\n";
                }

                parallelResults.clear();
                sequentialResults.clear();
            }
        }
    }

    return 0;
}
