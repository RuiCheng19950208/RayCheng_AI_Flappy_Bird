# Reference: https://www.youtube.com/watch?v=MMxFDaIOHsE&list=PLzMcBGfZo4-lwGZWXz5Qgta_YNX3_vLS2
import pygame
import os
import random
import pickle
import cv2
import pandas as pd
from pygame.locals import *
from sys import exit
from vector import Vector
import math
import operator
import numpy as np
import time
import csv
from hmmlearn import hmm





#--------------basic setting--------------------------

width=600
height=800
FLOOR = height+40
fps=120
AI_generation = 0
Pipe_born_x=1000
Pipe_born_interval=600


#----------color---------
WHITE=(255,255,255)
BLACK=(0,0,0)
BLUE=(255,0,0)
YELLOW=(255,255,0)
GREEN=(0,255,0)
RED=(0,0,255)
#-----------------------font---------------------------

font_name= os.path.join("simusun.ttc")

#------------------------------------------------------

class Player_bird(pygame.sprite.Sprite):
    """
    Bird class representing the flappy bird
    """
    def __init__(self,game, x, y):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :return: None
        """

        self.jump_vel = -6
        self.groups = game.all_sprites,game.player_bird_group
        pygame.sprite.Sprite.__init__(self,self.groups)
        self.game=game
        self.time_scale = self.game.time_scale
        self.ANIMATION_TIME = 5 * self.time_scale
        self.x = x
        self.y = y
        self.gravity_a = self.game.gravity_a
        self.vel = 0
        self.img_count = 0
        self.IMGS = self.game.bird_images
        self.image = self.IMGS[0]

        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centery = y
        self.rect.centerx = x


    def get_key(self):
        keys=pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            self.jump()
    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = self.jump_vel * self.time_scale

    def move(self):
        """
        make the bird move
        :return: None
        """
        # for downward acceleration
        self.vel += self.gravity_a
        self.y += self.vel

    def update(self):
        """
        draw the bird
        :param win: pygame window or surface
        :return: None
        """
        self.get_key()
        self.img_count += 1
        if self.y<=0:
            self.y=0
            self.vel=0
        if self.vel >10:
            self.vel=10

        # For animation of bird, loop through three images
        if self.img_count <= self.ANIMATION_TIME:
            self.image = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.image = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.image = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.image = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.image = self.IMGS[0]
            self.img_count = 0

        # so when bird is nose diving it isn't flapping

        # tilt the bird
        self.rect.y = self.y

        self.move()




class Pipe_bot(pygame.sprite.Sprite):
    """
    represents a pipe object
    """
    def __init__(self, game,x,y):
        self.groups = game.all_sprites, game.pipes_group
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.game=game
        self.vel = self.game.x_vel
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(self.game.pipe_img, False, True)
        self.image = self.game.pipe_img
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y
        self.passed = False

    def update(self):
        self.rect.x -= self.vel


class Pipe_top(pygame.sprite.Sprite):
    """
    represents a pipe object
    """
    def __init__(self, game,x,y):
        self.groups = game.all_sprites, game.pipes_group
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.game=game
        self.GAP = 300
        self.vel = self.game.x_vel
        self.image = pygame.transform.flip(self.game.pipe_img, False, True)
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y - self.GAP - self.image.get_height()
        self.passed = False

    def update(self):
        self.rect.x -= self.vel
        if self.rect.x<= self.game.player.x and self.passed ==False:
            self.passed =True
            self.game.random_next_pipe_heigh = random.randrange(height-150, height+150)
            self.add_pipe()

    def add_pipe(self):
        self.game.score +=1
        pipe_bot=Pipe_bot(self.game,self.rect.x+2*Pipe_born_interval,self.game.random_next_pipe_heigh)
        pipe_top=Pipe_top(self.game,self.rect.x+2*Pipe_born_interval,self.game.random_next_pipe_heigh)
        self.game.pipes.append(pipe_bot)
        self.game.pipes.append(pipe_top)
    def kill_self(self):
        if self.rect.x<-200:
            self.kill()






class Base(pygame.sprite.Sprite):
    """
    Represnts the moving floor of the game
    """
    def __init__(self, game, y):
        """
        Initialize the object
        :param y: int
        :return: None
        """
        self.groups = game.all_sprites, game.base_group
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.game=game
        self.image = self.game.base_img
        self.time_scale = self.game.time_scale
        self.vel = self.game.x_vel
        self.WIDTH = self.game.base_img.get_width()

        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.centery = y
        self.rect.centerx = (self.x1+self.x2)/2

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.vel
        self.x2 -= self.vel
        if self.x1 + self.x2 < 0:
            self.x1 = 0
            self.x2=self.WIDTH

    def update(self):
        """
        Draw the floor. This is two images that move together.
        :param win: the pygame surface/window
        :return: None
        """
        self.move()
        self.rect.centery = self.y
        self.rect.centerx = (self.x1 + self.x2) / 2


class Game():
    def __init__(self):
        # game window initialize
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Ray Cheng Flappy Bird with AI")
        self.clock = pygame.time.Clock()
        self.running =True
        self.font_name= pygame.font.match_font(font_name)
        self.snd_dir='sounds'
        self.img_dir='imgs'
        self.music_init(self.snd_dir)
        self.img_init(self.img_dir)
        self.AI_birds_num=50
        self.Born_location=[width/5,height/2]
        self.time_scale_list=[1,2,5,10]
        self.time_scale_index = 0
        self.time_scale = self.time_scale_list[self.time_scale_index]
        self.x_vel = 4 * self.time_scale
        self.gravity_a = 0.2 * self.time_scale
        self.score = 0
        self.random_next_pipe_heigh = random.randrange(height-150, height+150)

    def music_init(self,snd_dir):
        self.BGM=pygame.mixer.music.load(os.path.join(snd_dir,'game_music.wav'))
        self.GameOverSound = pygame.mixer.Sound(os.path.join(snd_dir, 'game_over.wav'))


    def img_init(self,img_dir):
        self.pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join(img_dir, "pipe.png")).convert_alpha())
        self.bg_img = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, "bg.png")).convert_alpha(), (600, 900))
        self.bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join(img_dir, "bird" + str(x) + ".png"))) for x in range(1, 4)]
        self.base_img = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, "base.png")).convert_alpha(),(2*width,300))

    # def loaddata(self):
    #     game_folder = os.path.dirname(__file__)
    #     self.map_data = []
    #     with open(os.path.join(game_folder,'MapEditor.txt'),'rt') as f:
    #         for line in f:
    #             self.map_data.append(line)

    def new(self):
        #restart
        self.all_sprites = pygame.sprite.Group()
        self.player_bird_group = pygame.sprite.Group()
        self.AI_bird_group = pygame.sprite.Group()
        self.pipes_group = pygame.sprite.Group()
        self.base_group = pygame.sprite.Group()

        self.player =  Player_bird(self,self.Born_location[0],self.Born_location[0])
        self.base = Base(self,FLOOR)
        self.pipes=[]
        self.pipes.append(Pipe_bot(self,Pipe_born_x,self.random_next_pipe_heigh))
        self.pipes.append(Pipe_top(self, Pipe_born_x,self.random_next_pipe_heigh))
        self.pipes.append(Pipe_bot(self, Pipe_born_interval + Pipe_born_x, self.random_next_pipe_heigh))
        self.pipes.append(Pipe_top(self, Pipe_born_interval + Pipe_born_x, self.random_next_pipe_heigh))



        # self.AI_birds = AI_bird(self, Player2Born[0], Player2Born[1])



        # for i in self.AI_birds_num:
        #     AI_bird(self, Player2Born[0], Player2Born[1])

        # for row,tiles in enumerate(self.map_data):
        #     for col,tile in enumerate(tiles):
        #         if tile=='2':
        #             Wall(self,col,row)


        self.run()

    def run(self):
        # game loop
        # pygame.mixer.music.play(loops=-1)
        self.playing =True
        self.score=0
        while self.playing:
            self.dt = self.clock.tick(fps)/1000
            self.events()
            self.update()
            self.draw()

    def update(self):
       #loop update
        self.all_sprites.update()

        # for b in self.bullets:
        #     if pygame.sprite.spritecollide(b, self.bricks, True):
        #         b.kill()
        #     if pygame.sprite.spritecollide(b, self.walls, False):
        #         b.kill()
        #     if pygame.sprite.spritecollide(b, self.enemy, True):
        #         b.kill()
        #         self.enemykill_sound.play()


        if pygame.sprite.spritecollide(self.player, self.base_group, True) or pygame.sprite.spritecollide(self.player, self.pipes_group, True) :
            self.player.kill()
            pygame.mixer.music.stop()
            self.GameOverSound.play()
            self.playing=False



    def events(self):
        #define trigger
        for event in pygame.event.get():
            if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:

                if self.playing:
                    self.playing = False

                self.running = False



    def draw(self):

        # draw in the window
        self.screen.fill(BLACK)
        self.screen.blit(self.bg_img, (0,0))
        self.all_sprites.draw(self.screen)
        self.draw_text_left_top('Author:Ray Cheng', 30, WHITE, 5, 0)
        self.draw_text_left_top('Score: '+str(self.score), 30, WHITE, 5, 32)
        pygame.display.flip()

    def draw_text(self,text,size,color,x,y):
        font=pygame.font.Font(self.font_name,size)
        text_surface = font.render(text,True,color)
        text_rect= text_surface.get_rect()
        text_rect.midtop=(x,y)
        self.screen.blit(text_surface,text_rect)

    def draw_text_left_top(self,text,size,color,x,y):
        font=pygame.font.Font(self.font_name,size)
        text_surface = font.render(text,True,color)
        text_rect= text_surface.get_rect()
        text_rect.topleft=(x,y)
        self.screen.blit(text_surface,text_rect)

    def show_start_screen(self):

        self.screen.fill(BLACK)
        self.draw_text('Ray Cheng Flappy Bird!',50,WHITE,width/2,height/4)
        self.draw_text('Press Q to start player mode game!', 30, WHITE, width / 2, (height / 4) + 140)
        self.draw_text('Press W to start AI training mode game!', 30, WHITE, width / 2, (height / 4) + 200)
        self.draw_text('Press E to start player vs AI mode game!', 30, WHITE, width / 2, (height / 4) + 260)
        pygame.display.flip()
        self.waitforkey()



    def show_gameover_screen(self):
        if not self.running:
            return

        self.screen.fill(BLACK)
        self.draw_text('Game Over',50, WHITE, width / 2, height / 4)
        self.draw_text('Press Q to start player mode game!', 30, WHITE, width / 2, (height / 4) + 140)
        self.draw_text('Press W to start AI training mode game!', 30, WHITE, width / 2, (height / 4) + 200)
        self.draw_text('Press E to start player vs AI mode game!', 30, WHITE, width / 2, (height / 4) + 260)
        pygame.display.flip()
        self.waitforkey()

    def waitforkey(self):
        waiting=True
        while waiting:
            self.clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:
                    waiting =False
                    self.running= False
                if  pygame.key.get_pressed()[pygame.K_q]:
                    waiting= False
                    pygame.mixer.music.play(-1)


    def get_image(self,img, x, y, w, h):
        image = pygame.Surface((w, h))
        image.blit(img, (0, 0), (x, y, w, h))
        # image=pygame.transform.scale(image,(w//2,h//2)) #If you want to resize
        return image



g=Game()
g.show_start_screen()

while g.running:
    g.new()
    g.show_gameover_screen()

pygame.quit()

