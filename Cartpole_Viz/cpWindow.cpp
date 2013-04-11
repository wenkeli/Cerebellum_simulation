#include "cpWindow.hpp"
#include <sstream>
#include <iostream>

using namespace std;

const SDL_Rect CPWindow::blue_back = {212,148,60,58};
const SDL_Rect CPWindow::blue_reverse = {74,148,60,58};
const SDL_Rect CPWindow::blue_play = {106,50,60,58};
const SDL_Rect CPWindow::blue_pause = {177,50,60,58};
const SDL_Rect CPWindow::blue_ff   = {142,148,60,58};
const SDL_Rect CPWindow::blue_next = {279,148,60,58};

const SDL_Rect CPWindow::purple_reverse = {360,148,60,58};
const SDL_Rect CPWindow::purple_back = {497,148,60,58};
const SDL_Rect CPWindow::purple_ff = {428,148,60,58};
const SDL_Rect CPWindow::purple_next = {565,148,60,58};
const SDL_Rect CPWindow::purple_play = {391,50,60,58};
const SDL_Rect CPWindow::purple_pause = {463,50,60,58};

const SDL_Rect CPWindow::speedup = {230,283,65,65};
const SDL_Rect CPWindow::slowdown = {160,283,65,65};

Button::Button(int x, int y, int w, int h, 
               const SDL_Rect* _clip, const SDL_Rect* _mousedown_clip,
               const SDL_Rect* _thirdClip, const SDL_Rect* _fourthClip):
    pressed(false), inverted(false)
{
    box.x = x; box.y = y; box.w = w; box.h = h;
    clip1 = (SDL_Rect*) _clip;
    clip2 = (SDL_Rect*) _mousedown_clip;
    clip3 = (SDL_Rect*) _thirdClip;
    clip4 = (SDL_Rect*) _fourthClip;
}

void Button::setSurfaces(SDL_Surface* _screen, SDL_Surface* _buttonSheet)
{
    screen = _screen;
    buttonSheet = _buttonSheet;
}

bool Button::handle_events(const SDL_Event event)
{
    bool active = false;

    //The mouse offsets
    int x = 0, y = 0;

    //If the mouse moved
    if(event.type == SDL_MOUSEMOTION && pressed) 
    {
        x = event.motion.x;
        y = event.motion.y;

        if( ( x > box.x ) && ( x < box.x + box.w ) && ( y > box.y ) && ( y < box.y + box.h ) )
            ;
        else
            pressed = false;
    }

    if( event.type == SDL_MOUSEBUTTONDOWN ) {
        if( event.button.button == SDL_BUTTON_LEFT ) {
            //Get the mouse offsets
            x = event.button.x;
            y = event.button.y;

            //If the mouse is over the button
            if( ( x > box.x ) && ( x < box.x + box.w ) && ( y > box.y ) && ( y < box.y + box.h ) )
                pressed = true;
        }
    }

    if( event.type == SDL_MOUSEBUTTONUP ) {
        if( event.button.button == SDL_BUTTON_LEFT ) {
            //Get the mouse offsets
            x = event.button.x;
            y = event.button.y;

            //If the mouse is over the button
            if( ( x > box.x ) && ( x < box.x + box.w ) && ( y > box.y ) && ( y < box.y + box.h ) ) {
                if (pressed) {
                    active = true;
                    inverted = !inverted;
                }
                pressed = false;
            }
        }
    }
    return active;
}

void Button::show()
{
    //Holds offsets
    SDL_Rect offset;

    //Get offsets
    offset.x = box.x;
    offset.y = box.y;

    //Blit
    if (inverted && clip3 != NULL && clip4 != NULL) {
        if (pressed)
            SDL_BlitSurface(buttonSheet, clip4, screen, &offset);
        else
            SDL_BlitSurface(buttonSheet, clip3, screen, &offset);
    } else {
        if (pressed)
            SDL_BlitSurface(buttonSheet, clip2, screen, &offset);
        else
            SDL_BlitSurface(buttonSheet, clip1, screen, &offset);
    }        
}


void CPWindow::SDL_Mainloop()
{
    while (1) {
        SDL_WaitEvent(&event);
        if (event.type == SDL_QUIT)
            exit(0);

        int mediaEvent = -1;

        if (backButton.handle_events(event))
            mediaEvent = PREVIOUS;
        if (reverseButton.handle_events(event))
            mediaEvent = BACK;
        if (playButton.handle_events(event))
            mediaEvent = PLAYPAUSE;
        if (ffButton.handle_events(event))
            mediaEvent = FF;
        if (nextButton.handle_events(event))
            mediaEvent = NEXT;
        if (slowdownButton.handle_events(event))
            mediaEvent = SLOWDOWN;
        if (speedupButton.handle_events(event))
            mediaEvent = SPEEDUP;

        renderScreen();

        if (mediaEvent >= 0)
            for (int i=0; i<handlers.size(); ++i)
                handlers[i]->handleMediaEvent(mediaEvent);
    }
}


CPWindow::CPWindow(float _tracklen, float _polelen, float _leftAngBound, float _rightAngBound,
                   float _lower_cartWidth) :
    tracklen(_tracklen), polelen(_polelen), leftAngBound(_leftAngBound), rightAngBound(_rightAngBound),
    lower_cartWidth(_lower_cartWidth),
    cartpos(0), cartvel(0), poleang(0), polevel(0), lower_cartpos(0), lower_cartvel(0),
    lower_force(0), lower_target(0), forceLeft(0), forceRight(0), errorLeft(0), errorRight(0),
    timeAloft(0), trialNum(0), cycle(0), playspeed(1),
    bleedLeft(0), bleedRight(0), 
    screenWidth(640), screenHeight(400), screenBPP(32),
    backButton(0,340,60,58,&blue_back,&purple_back),
    reverseButton(60,340,60,58,&blue_reverse,&purple_reverse),
    playButton(120,340,60,58,&blue_pause,&purple_pause,&blue_play,&purple_play),
    ffButton(180,340,60,58,&blue_ff,&purple_ff),
    nextButton(240,340,60,58,&blue_next,&purple_next),
    slowdownButton(340,340,60,58,&slowdown,&slowdown),
    speedupButton(405,340,60,58,&speedup,&speedup)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        cerr << "Problem in SDL_Init" << endl;
        exit(1);
    }

    // Initialize SDL_ttf library
    if (TTF_Init() != 0) {
        cerr << "TTF_Init() Failed: " << TTF_GetError() << endl;
        SDL_Quit();
        exit(1);
    }

    // Load a font
    font = TTF_OpenFont("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 14);
    if (font == NULL) {
        cerr << "TTF_OpenFont() Failed: " << TTF_GetError() << endl;
        TTF_Quit();
        SDL_Quit();
        exit(1);
    }

    screen = SDL_SetVideoMode(screenWidth, screenHeight, 0, SDL_HWSURFACE | SDL_DOUBLEBUF);
    SDL_WM_SetCaption("Cartpole Viz", 0);

    // Load the button image
    buttonSheet = load_image("buttons.jpg");

    backButton.setSurfaces(screen,buttonSheet);
    reverseButton.setSurfaces(screen,buttonSheet);
    playButton.setSurfaces(screen,buttonSheet);
    ffButton.setSurfaces(screen,buttonSheet);
    nextButton.setSurfaces(screen,buttonSheet);    
    slowdownButton.setSurfaces(screen,buttonSheet);
    speedupButton.setSurfaces(screen,buttonSheet);        

    // Create a new thread to run the GL stuff
    pthread_create(&thread, NULL, &CPWindow::thread_helper, this);
};

CPWindow::~CPWindow() {
    pthread_cancel(thread);
    
    SDL_FreeSurface(buttonSheet);
    SDL_Quit();
};


SDL_Surface* CPWindow::load_image( std::string filename )
{
    SDL_Surface* loadedImage = NULL;
    SDL_Surface* optimizedImage = NULL;
    loadedImage = IMG_Load(filename.c_str());
    if(loadedImage != NULL)
    {
        //Create an optimized surface
        optimizedImage = SDL_DisplayFormat( loadedImage );
        SDL_FreeSurface(loadedImage);
        //If the surface was optimized
        if(optimizedImage != NULL)
        {
            //Color key surface
            SDL_SetColorKey( optimizedImage, SDL_SRCCOLORKEY, SDL_MapRGB( optimizedImage->format, 0, 0xFF, 0xFF ) );
        }
    }

    //Return the optimized surface
    return optimizedImage;
}


void CPWindow::drawCartpole(float cartpos, float cartvel, float poleang, float polevel,
                            float lower_cartpos, float lower_cartvel, float lower_force, float lower_target,
                            float forceLeft, float forceRight, bool errorLeft, bool errorRight,
                            int timeAloft, int trialNum, int cycle, float playspeed) {
    this->cartpos = cartpos;
    this->cartvel = cartvel;
    this->poleang = poleang;
    this->polevel = polevel;
    this->lower_cartpos = lower_cartpos;
    this->lower_cartvel = lower_cartvel;
    this->lower_force = lower_force;
    this->lower_target = lower_target;
    this->forceLeft = forceLeft;
    this->forceRight = forceRight;
    this->errorLeft = errorLeft;
    this->errorRight = errorRight;
    this->timeAloft = timeAloft;
    this->trialNum = trialNum;
    this->cycle = cycle;
    this->playspeed = playspeed;

    // Push a user event which will be picked up by our rendering thread
    SDL_Event e;
    e.type = SDL_USEREVENT;
    SDL_PushEvent(&event);
}
    
void CPWindow::renderScreen() {
    SDL_FillRect(screen, NULL, SDL_MapRGB(screen->format, 20, 20, 20));
    
    // Width & height of sub-area of screen we are drawing on
    int width = screenWidth;           // 400
    int height = 0.8 * screenHeight;   // 320

    // Track
    int trackLenPix = .8 * width;
    int trackStartX = (width - trackLenPix)/2;
    int trackEndX   = width - (width - trackLenPix)/2;
    int trackY      = .9 * height;
    hlineRGBA(screen, trackStartX, trackEndX, trackY, 255,255,255, 255);
    
    float effectiveTracklen = tracklen;
    float effectiveLowerCartPos = lower_cartpos;
    if (tracklen >= 1e30) {
        effectiveTracklen = 40.0f;
        //lower_target = lower_target - lower_cartpos;
        effectiveLowerCartPos = 0.0f;
    }

    float effectiveLowerCartwidth = lower_cartWidth;
    if (lower_cartWidth >= 1e30) {
        effectiveLowerCartwidth = 40.0f;
        cartpos = 0.0f;
        effectiveLowerCartPos = 0.0f;
    }
    
    float meters2pix = trackLenPix / (float) effectiveTracklen;

    // Draw the lower cart
    float lowerDistAlongTrack = effectiveTracklen/2.0f + effectiveLowerCartPos; 
    int lower_cartlen_pix = (effectiveLowerCartwidth / effectiveTracklen) * trackLenPix;
    int lower_cartpos_pix = (lowerDistAlongTrack / effectiveTracklen) * trackLenPix + trackStartX;
    int lower_target_pix  = (((lower_target - lower_cartpos) + lowerDistAlongTrack) / effectiveTracklen) * trackLenPix + trackStartX;
    int lowerCartUpperDeckHeight = .85*height;
    int lowerCartLowerDeckHeight = .88*height;
    boxRGBA(screen,
            lower_cartpos_pix + lower_cartlen_pix/2, lowerCartUpperDeckHeight, // Top right (x,y)
            lower_cartpos_pix - lower_cartlen_pix/2, lowerCartLowerDeckHeight, // Bottom left (x,y)
            255,143,0, 255); // Colors and alpha
    // Trigon showing the center of the lower cart
    filledTrigonRGBA(screen,
                     lower_cartpos_pix, lowerCartLowerDeckHeight,
                     lower_cartpos_pix-2, lowerCartUpperDeckHeight+1,
                     lower_cartpos_pix+2, lowerCartUpperDeckHeight+1,
                     60, 125, 196, 255);

    // Draw the Upper Cart
    int cartlen = 5;
    float distAlongTrack = effectiveTracklen / 2.0f + cartpos - effectiveLowerCartPos;
    int cartlen_pix = (cartlen / effectiveTracklen) * trackLenPix;
    int cartpos_pix = (distAlongTrack / effectiveTracklen) * trackLenPix + trackStartX;
    int upperCartUpperDeckHeight = .81*height;
    int upperCartLowerDeckHeight = .84*height;
    boxRGBA(screen,
            cartpos_pix + cartlen_pix/2, upperCartUpperDeckHeight, // Top right (x,y)
            cartpos_pix - cartlen_pix/2, upperCartLowerDeckHeight, // Bottom left (x,y)
            249, 249, 207, 255); // Colors and alpha

    // Draw the left/right pole angle bounds
    int add_x = polelen*sin(leftAngBound) * meters2pix;
    int add_y = cos(leftAngBound)*polelen * meters2pix;
    lineRGBA(screen, 
             cartpos_pix, upperCartUpperDeckHeight,
             cartpos_pix+add_x, upperCartUpperDeckHeight - abs(add_y),
             255,0,0, 100); // Color and alpha

    // Draw the left/right bounds
    add_x = polelen*sin(rightAngBound) * meters2pix;
    add_y = cos(rightAngBound)*polelen * meters2pix;
    lineRGBA(screen, 
             cartpos_pix, upperCartUpperDeckHeight,
             cartpos_pix+add_x, upperCartUpperDeckHeight - abs(add_y),
             255,0,0, 100); // Color and alpha

    // Draw the pole
    add_x = polelen*sin(poleang) * meters2pix;
    add_y = cos(poleang)*polelen * meters2pix;
    lineRGBA(screen, 
             cartpos_pix, upperCartUpperDeckHeight,
             cartpos_pix-add_x, upperCartUpperDeckHeight - abs(add_y),
             249, 249, 207, 255); // Color and alpha

    // Draw Upper Cart forces
    if (forceLeft > 0)
        boxRGBA(screen,
                cartpos_pix,upperCartUpperDeckHeight+2,// top Right (x1,y1)
                cartpos_pix-(trackLenPix/2)*forceLeft, upperCartLowerDeckHeight-2, // bottom left (x2,y2)
                72, 217, 225, 128);

    if (forceRight > 0)
        boxRGBA(screen,
                cartpos_pix+(trackLenPix/2)*forceRight+1,upperCartUpperDeckHeight+2,// top Right (x1,y1)
                cartpos_pix+1, upperCartLowerDeckHeight-2, // bottom left (x2,y2)
                72, 141, 225, 128);

    float netforce = forceRight - forceLeft;

    if (netforce > 0) {
        boxRGBA(screen,
                cartpos_pix+(trackLenPix/2)*netforce+1,upperCartUpperDeckHeight+2,// top Right (x1,y1)
                cartpos_pix+1, upperCartLowerDeckHeight-2, // bottom left (x2,y2)
                225, 72, 217, 255);
    } else if (netforce < 0) {
        boxRGBA(screen,
                cartpos_pix,upperCartUpperDeckHeight+2,// top Right (x1,y1)
                cartpos_pix+(trackLenPix/2)*netforce, upperCartLowerDeckHeight-2, // bottom left (x2,y2)
                225, 72, 217, 255);
    }

    // Draw lower cart forces
    if (lower_force > 0) {
        boxRGBA(screen,
                lower_cartpos_pix+(trackLenPix/2)*lower_force,trackY+5,// top Right (x1,y1)
                lower_cartpos_pix, trackY + 10, // bottom left (x2,y2)
                255,0,255,255);
    } else if (lower_force < 0) {
        boxRGBA(screen,
                lower_cartpos_pix,trackY+5,// top Right (x1,y1)
                lower_cartpos_pix+(trackLenPix/2)*lower_force, trackY+10, // bottom left (x2,y2)
                160,32,240,255);
    }

    // Draw destination triangle
    filledTrigonRGBA(screen,
                     lower_target_pix, lowerCartLowerDeckHeight+2,
                     lower_target_pix-2, lowerCartLowerDeckHeight+8,
                     lower_target_pix+2, lowerCartLowerDeckHeight+8,
                     253,20,145, 255);

    // Draw Errors
    bleedRight *= .9;
    bleedLeft *= .9;
    
    if (errorRight)
        bleedRight = 255;
    if (errorLeft)
        bleedLeft = 255;

    //circleRGBA(screen,trackStartX,height*.5,20,255,0,0,255);

    filledCircleRGBA(screen,trackStartX,height*.5,20,255,0,0,bleedLeft);
    filledCircleRGBA(screen,trackEndX,height*.5,20,255,0,0,bleedRight);

    stringstream ss;
    SDL_Rect dest;
    
    ss.str("");
    ss << "Trial Num: " << trialNum;
    dest = {0,0,width,height};
    drawText(ss.str(), dest);

    ss.str("");
    ss << "TimeAloft: " << timeAloft;
    dest = {175,0,width,height};
    drawText(ss.str(), dest);

    ss.str("");
    ss << "Cycle: " << cycle;
    dest = {350,0,width,height};
    drawText(ss.str(), dest);

    // Write pole ang text
    ss.str("");
    ss << "PoleAng: " << poleang;
    dest = {0,15,width,height};
    drawText(ss.str(), dest);

    // Pole Vel Text
    ss.str("");
    ss << "PoleVel: " << polevel;
    dest = {0,30,width,height};
    drawText(ss.str(), dest);

    // Write cart pos text
    ss.str("");
    ss << "RelCartPos: " << (cartpos - lower_cartpos);
    dest = {175,15,width,height};
    drawText(ss.str(), dest);

    ss.str("");
    ss << "RelCartVel: " << (cartvel - lower_cartvel);
    dest = {175,30,width,height};
    drawText(ss.str(), dest);

    // Lower Cart Pos Text
    ss.str("");
    ss << "LowerCartPos: " << lower_cartpos;
    dest = {350,15,width,height};
    drawText(ss.str(), dest);

    ss.str("");
    ss << "LowerCartVel: " << lower_cartvel;
    dest = {350,30,width,height};
    drawText(ss.str(), dest);


    ss.str("");
    ss << "Speed: " << playspeed << "x";
    dest = {475,360,width,height};
    drawText(ss.str(), dest);

    ss.str("");
    ss << "ErrL";
    dest = {trackStartX-13,height*.5-10,width,height};
    drawText(ss.str(), dest);

    ss.str("");
    ss << "ErrR";
    dest = {trackEndX-13,height*.5-10,width,height};
    drawText(ss.str(), dest);

    //Show the buttons
    backButton.show();
    reverseButton.show();
    playButton.show();
    ffButton.show();
    nextButton.show();
    speedupButton.show();
    slowdownButton.show();

    SDL_Flip(screen);
    //SDL_Delay(1);
    SDL_FreeSurface(screen);
};

void CPWindow::drawText(string s, SDL_Rect& dest) {
    SDL_Color text_color = {255, 255, 255};
    SDL_Surface* text = TTF_RenderText_Blended(font,s.c_str(),text_color);

    if (text == NULL) {
        cerr << "TTF_RenderText_Solid() Failed: " << TTF_GetError() << endl;
        TTF_Quit();
        SDL_Quit();
        exit(1); 
    }

    SDL_BlitSurface(text, NULL, screen, &dest);
    SDL_FreeSurface(text);
};
