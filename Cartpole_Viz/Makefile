all: logviz 

logviz:
	g++ -c -o cpWindow.o cpWindow.cpp
	g++ -c -o logReader.o logReader.cpp
	g++ -o CartPoleViz logReader.o cpWindow.o -l SDL -l SDL_gfx -l SDL_ttf -l SDL_image 

clean:
	rm -f *.o CartPoleViz