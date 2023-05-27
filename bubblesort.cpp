#include <iostream>
#include <omp.h>

using namespace std;

// Function to perform bubble sort on a subarray
void bubbleSort(int arr[], int left, int right) {
    for (int i = left; i <= right; i++) {
        for (int j = left; j <= right - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap adjacent elements if they are in the wrong order
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Function to print the elements of an array
void printArray(int arr[],int size) 
{
    for (int i = 0; i < size; i++)
        cout << arr[i] << " "<<endl;
}

int main() 
{
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    int* arr = new int[n];

    cout << "Enter the elements: ";
    for (int i = 0; i < n; i++)
        cin >> arr[i];

    // Perform parallel bubble sort
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < n - 1; i++) {
                #pragma omp task
                {
                    bubbleSort(arr, 0, n - 1 - i);
                }
            }
        }
    }

    cout << "Sorted array: ";
    printArray(arr,n);

    return 0;
}
