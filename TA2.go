package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type Empleado struct {
	salary float64
	gender int64
	age    int64
	PhD    int64
}

var (
	empleados []Empleado
)

func setRango() [2]int { //setea el rango del dataset con el cual se va a entrenar
	values := [2]int{}
	r := bufio.NewReader(os.Stdin)
	str, _ := r.ReadString('\n')
	left, err := strconv.Atoi(strings.TrimSpace(str))
	str, _ = r.ReadString('\n')
	right, err2 := strconv.Atoi(strings.TrimSpace(str))
	if err != nil || err2 != nil {
		fmt.Println("Invalid output")
		return [2]int{0, 0}
	}
	fmt.Println("You entered: left ", left, "rig", right)
	values[0] = left
	values[1] = right
	return values
}
func leerDatos() {

	//abrir csv
	csvFile, err := os.Open("Datasets/Salary.csv")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened CSV file")
	defer csvFile.Close()
	// Leer y descartar la primera fila
	reader := csv.NewReader(csvFile)
	_, err = reader.Read()
	if err != nil {
		fmt.Println("Error in read the first file:", err)
		return
	}
	//recuperar informacion del dataset
	csvLines, err := reader.ReadAll()

	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf("Number of rows in CSV file: %d\n", len(csvLines))
	v := setRango()
	left := v[0]
	right := v[1]
	fmt.Printf("left: ", left)
	fmt.Printf("right: ", right)
	//procesar dataset
	for i, line := range csvLines {
		if i >= left && i < right {

			salary, err := strconv.ParseFloat(line[0], 64)
			gender, err1 := strconv.Atoi(line[1])
			age, err2 := strconv.Atoi(line[2])
			PhD, err3 := strconv.Atoi(line[3])
			if err != nil || err1 != nil || err2 != nil || err3 != nil {
				fmt.Printf("Error in read files")
				return
			}
			emp := Empleado{
				salary: salary,
				gender: int64(gender),
				age:    int64(age),
				PhD:    int64(PhD),
			}
			empleados = append(empleados, emp)
			//fmt.Printf("new employee!\n")
		}
		//fmt.Println(height, weitht)
	}
	fmt.Printf("Number of employees: %d\n", len(empleados))

}

func matrices() []*mat.Dense {
	if len(empleados) == 0 {
		// Manejar el caso de que empleados esté vacío
		return []*mat.Dense{}
	}
	X := mat.NewDense(len(empleados), 2, nil)
	Y := mat.NewDense(len(empleados), 1, nil)

	for i, v := range empleados {
		if i >= 0 && i < len(empleados) {
			X.Set(i, 0, v.salary)
			X.Set(i, 1, float64(v.age))
			Y.Set(i, 0, float64(v.gender))
		}
	}
	matrices := []*mat.Dense{X, Y}
	return matrices
}

type DenseMatrix struct {
	*mat.Dense
}

func (m *DenseMatrix) At(i, j int) float64 {
	return m.Dense.At(i, j)
}

func (m *DenseMatrix) Dims() (r, c int) {
	return m.Dense.RawMatrix().Rows, m.Dense.RawMatrix().Cols
}
func regresion_lineal_multiple_simple() []*DenseMatrix {
	matrices := matrices()
	X := DenseMatrix{matrices[0]}
	Y := DenseMatrix{matrices[1]}
	Wd := mat.NewDense(X.RawMatrix().Cols, 1, nil)
	W := DenseMatrix{Wd}
	Xt := X.T()

	//calcular XtX
	var XtXResult mat.Dense
	XtXResult.Mul(Xt, X.Dense)

	//calcular XtY
	var XtYResult mat.Dense
	XtYResult.Mul(Xt, Y.Dense)

	// Usar los resultados de XtX y XtY
	var XtxInverse mat.Dense
	err := XtxInverse.Inverse(&XtXResult)
	if err != nil {
		log.Fatalf("Xtx is not invertible: %v", err)
	}

	W.Mul(&XtxInverse, &XtYResult)
	resultados := []*DenseMatrix{&X, &Y, &W}
	return resultados
}

func regresion_lineal_multiple() []*DenseMatrix {
	matrices := matrices()
	X := DenseMatrix{matrices[0]}
	Y := DenseMatrix{matrices[1]}
	Wd := mat.NewDense(X.RawMatrix().Cols, 1, nil)
	W := DenseMatrix{Wd}
	Xt := X.T()

	// Crear el wait group
	var wg sync.WaitGroup

	// Empezar el goroutine para calcular XtX
	var XtXResult mat.Dense
	wg.Add(1)
	go func() {
		defer wg.Done()
		XtXResult.Mul(Xt, X.Dense)
	}()

	// Empezar el goroutine para calcular XtY
	var XtYResult mat.Dense
	wg.Add(1)
	go func() {
		defer wg.Done()
		XtYResult.Mul(Xt, Y.Dense)
	}()

	// Esperar los goroutines para terminar
	wg.Wait()

	// Usar los resultados de XtX y XtY
	var XtxInverse mat.Dense
	err := XtxInverse.Inverse(&XtXResult)
	if err != nil {
		log.Fatalf("Xtx is not invertible: %v", err)
	}

	W.Mul(&XtxInverse, &XtYResult)
	resultados := []*DenseMatrix{&X, &Y, &W}
	return resultados
}
func regresion_polinomial(degree int) []*DenseMatrix {
	if degree >= len(empleados) {
		fmt.Println("No se ingreso una elevación correcta\n")
		return []*DenseMatrix{}
	}
	matrices := matrices()
	X := DenseMatrix{matrices[0]}
	Y := DenseMatrix{matrices[1]}
	Wd := mat.NewDense(degree+1, 1, nil)
	W := DenseMatrix{Wd}

	Xp := mat.NewDense(X.RawMatrix().Rows, degree+1, nil)
	Xpp := DenseMatrix{Xp}
	// Optimización del cálculo de potencias
	for i := 0; i < X.RawMatrix().Rows; i++ {
		xValue := X.At(i, 0) // Obtenemos el valor actual de X
		power := 1.0         // Comenzamos con x^0, que es 1

		for j := 0; j <= degree; j++ {
			Xpp.Set(i, j, power)
			power *= xValue // Calculamos la siguiente potencia multiplicando el valor actual
		}
	}

	Xt := Xpp.T()

	// Crear el wait group
	var wg sync.WaitGroup

	// Empezar el goroutine para calcular XtX
	var XtXResult mat.Dense
	wg.Add(1)
	go func() {
		defer wg.Done()
		XtXResult.Mul(Xt, Xpp.Dense)
	}()

	// Empezar el goroutine para calcular XtY
	var XtYResult mat.Dense
	wg.Add(1)
	go func() {
		defer wg.Done()
		XtYResult.Mul(Xt, Y.Dense)
	}()

	// Esperar los goroutines para terminar
	wg.Wait()

	// Usar los resultados de XtX y XtY
	var XtxInverse mat.Dense
	err := XtxInverse.Inverse(&XtXResult)
	if err != nil {
		log.Fatalf("Xtx is not invertible: %v", err)
	}

	W.Mul(&XtxInverse, &XtYResult)
	resultados := []*DenseMatrix{&Xpp, &Y, &W}
	return resultados
}

func imprimirMatrices(matrices []*DenseMatrix) {
	fmt.Printf("Y = %v\n\n", mat.Formatted(matrices[1], mat.Prefix("    "), mat.Squeeze()))
	fmt.Printf("X = %v\n\n", mat.Formatted(matrices[0], mat.Prefix("    "), mat.Squeeze()))
	fmt.Printf("W = %v\n\n", mat.Formatted(matrices[2], mat.Prefix("    "), mat.Squeeze()))
}

func RMSE(matrices []*DenseMatrix) {
	chi := make(chan int, len(empleados))
	e := make(chan float64, len(empleados))

	X := matrices[0]
	Y := matrices[1]
	W := matrices[2]

	Ypreditc := mat.NewDense(X.RawMatrix().Rows, Y.RawMatrix().Cols, nil)
	Ypreditc.Mul(X.Dense, W.Dense)
	var wg sync.WaitGroup
	wg.Add(len(empleados))
	var E float64 = 0
	for i := 0; i < len(empleados); i++ {
		chi <- i
		go func() {
			p := <-chi
			e <- math.Pow(Ypreditc.At(p, 0)-Y.At(p, 0), 2)
			wg.Done()
		}()
		E += <-e

	}
	wg.Wait()
	rmse := math.Sqrt(E / float64(len(empleados)))
	fmt.Println("\nEl error cuadrático medio es de ±", fmt.Sprintf("%.2f", rmse), "\n")
}

func main() {
	leerDatos()
	resultados := regresion_lineal_multiple()
	imprimirMatrices(resultados)
	RMSE(resultados)
	resultados2 := regresion_lineal_multiple_simple()
	imprimirMatrices(resultados2)
	RMSE(resultados2)
	resultados3 := regresion_polinomial(1)
	imprimirMatrices(resultados3)
	RMSE(resultados3)
}
