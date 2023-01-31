package main

import (
	"fmt"
	"sync"
	"time"
)

var set = make(map[int]bool, 0)
var m sync.Mutex

func test(stop chan bool) {
	fmt.Println("test start!")
	time.Sleep(time.Second)
	fmt.Println("test finish!")
	//stop <- true
}

func hh(stop chan bool) {
	fmt.Println("start hh")
	go test(stop)
	<-stop
	fmt.Println("stop hh")
}

func main() {
	stop := make(chan bool)
	go hh(stop)
	time.Sleep(time.Millisecond * 100)
	stop <- false

	time.Sleep(time.Second * 10)

	//fmt.Println("run")
	//a := area{dis: 1}
	//fmt.Println(a)
	//b := test(a)
	//fmt.Println(a)
	//fmt.Println(b)

	//var t string = "zpltys"
	//fmt.Println(t[len(t)-1] == 'f')
	//fmt.Println(t + "/")

	//a := []int{1, 2, 3, 4, 5, 6}
	//
	//for i := 0; i < len(a); i++ {
	//	if a[i] == 6 {
	//		a = append(a[:i], a[i+1:]...)
	//	}
	//}
	//fmt.Println(a)
	//fmt.Println(a[4:])

	//m := make(map[int][]int)
	//
	//m[0] = []int{2, 4, 76, 345, 32}
	////fmt.Println(m[0][1:3])
	//m[1] = m[0][1:3]
	//m[1][2] = 45
	//m[1] = append(m[1], 35)
	//fmt.Println(m)
	//
	////a := []int{3, 4, 76}
	//for _, v := range m[1] {
	//	fmt.Println(v)
	//}
	//
	//fmt.Println(m[1])
	//
	//if _, ok := m[0]; ok {
	//	fmt.Println("ok")
	//} else {
	//	fmt.Println("NO!")
	//}

}
