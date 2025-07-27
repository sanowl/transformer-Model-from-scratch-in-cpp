#include "matrix.h"
#include <iostream>

int main() {
    try {
        std::cout << "Testing Matrix creation..." << std::endl;
        Matrix m1(2, 3);
        m1.fill(1.0f);
        
        Matrix m2(2, 3);
        m2.fill(2.0f);
        
        std::cout << "Testing Matrix addition..." << std::endl;
        Matrix result = m1 + m2;
        
        std::cout << "Matrix addition successful!" << std::endl;
        std::cout << "Result matrix (should be all 3.0):" << std::endl;
        result.print();
        
        std::cout << "Testing Matrix multiplication..." << std::endl;
        Matrix a(2, 3);
        a.fill(1.0f);
        Matrix b(3, 2);
        b.fill(2.0f);
        
        Matrix c = a * b;
        std::cout << "Matrix multiplication successful!" << std::endl;
        std::cout << "Result matrix (should be all 6.0):" << std::endl;
        c.print();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}