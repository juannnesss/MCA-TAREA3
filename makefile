#### Corre la simulacion con N=1000 y epsilon=0.01
run : initial.txt final.txt
	gcc n_body.c -lm
	./a.out 200 0.1

#### Crea el archivo (vacio) con datos iniciales
initial.txt : 
	touch $@
#### Crea el archivo (vacio) con datos finales
final.txt :
	touch $@
#### Elimina archivos
clean :
	rm *.txt *.out

