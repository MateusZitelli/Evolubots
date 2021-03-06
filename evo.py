'''    Evolubots: a simple artificial life simulator.
Copyright (C) 2010  Mateus Zitelli Dantas (zitellimateus@gmail.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''


from math import *
from random import *
import sys
from time import *
from threading import Thread

import pygame
from pygame.locals import *
from pyfann import *
from pylab import *

def create_network(inputs, outputs):
    connection_rate = 1
    learning_rate = 0.7
    ann = libfann.neural_net()
    #Best configuration to fast learn and a small error
    number_hidden = (inputs + outputs) * 2 - 1
    ann.create_sparse_array(connection_rate, 
                            (inputs, number_hidden, outputs))
    ann.set_learning_rate(learning_rate)
    #Internal and ootputs sigmoid functions
    ann.set_activation_function_hidden(libfann.SIGMOID)
    ann.set_activation_function_output(libfann.SIGMOID)
    return ann

def primary_dad(bot):
    if bot.dad:
       return primary_dad(bot.dad)
    else:
       return bot

def draw_lines_between_bots(surface, bot, blist, bots_verified):
    for relative in blist:
        if not (relative, bot) in bots_verified or not (bot,
                relative) in bots_verified:
            pygame.draw.aaline(surface, (0,0,255), bot.pos, relative.pos)
            bots_verified.append((relative,bot))
            draw_lines_between_bots(surface, relative, relative.relatives,
                                    bots_verified)

def draw_genealogic_tree(surface, bot):
    verified = []
    for rela in bot.relatives:
        draw_lines_between_bots(surface, rela, rela.relatives, verified)

def bots_distance(b1, b2):
    return sqrt((b1.pos[0] - b2.pos[0]) ** 2 + (b1.pos[1] - b2.pos[1]) ** 2)

class Grass:
    """field of grass"""


    def __init__(self, width, heigth, surface, max_val_each_field = 100,
                 speed_of_obtention = 20, speed_of_growing = 0.5,
                 initial_size = 0):
        """
        width, heigth -> Number of fields on grass field.
        surface -> The surface that the grass will be.
        max_val_each_field -> Max quantity of food in eachfield.
        speed_of_obtention -> Quantity of food that the Bot will obtain in each
        turn.
        speed_of_growing -> Quantity of food that will grow up in each turn.
        initial_size -> Initial quantity of food in each field."""
        self.size = (width, heigth)
        self.screen = surface.get_size()
        self.surface = surface
        self.max_capacity = max_val_each_field
        self.facility_of_obtention = speed_of_obtention
        self.speed_of_growing = speed_of_growing
        self.initial_size = initial_size
        self.razao_x = self.size[0] / float(self.screen[0])
        self.razao_y = self.size[1] / float(self.screen[1])
        self.grid = []
        for x in range(width):
            self.grid.append([self.initial_size] * heigth)

    def grow(self):
        """Increase the quantity of food in each field"""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] < self.max_capacity:
                    self.grid[i][j] += self.speed_of_growing

    def eat(self, x, y):
        """Return the quantity of foof availabe in each field
                to the bot eat"""
        razao_x = self.size[0] / float(self.screen[0])
        razao_y = self.size[1] / float(self.screen[1])
        pos_x = int(x * razao_x)
        pos_y = int(y * razao_y)
        if self.grid[pos_x][pos_y] >= self.facility_of_obtention:
            self.grid[pos_x][pos_y] -= self.facility_of_obtention
            retired = self.facility_of_obtention
        else:
            retired = self.grid[pos_x][pos_y]
            self.grid[pos_x][pos_y] = 0
        return retired

    def draw(self):
        """Draw the grass in the surface"""
        razao_x = float(self.screen[0]) / self.size[0]
        razao_y = float(self.screen[1]) / self.size[1]
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                green = (self.grid[i][j] / float(self.max_capacity)) * 255
                if green > 255:
                    green = 255
                color = (255 - green , 255 , 255 - green)
                ipos = (i * razao_x, j * razao_y)
                fpos = ((i + 1) * razao_x, (j + 1) * razao_y)
                rect = pygame.Rect(ipos[0], ipos[1], fpos[0], fpos[1])
                self.surface.fill(color, rect)

    def get_food_quantity(self, x, y):
        pos_x = int(x * self.razao_x)
        pos_y = int(y * self.razao_y)
        return self.grid[pos_x - 1][pos_y - 1] / float(self.max_capacity)

class Bot(object):


    def __init__(self, x, y, surface, bots_list, grass, energy = 500,
                 variability_tax = 10, genetic_code = None,train_turns = 100):
        self.pos = [x, y]
        self.angle = radians(randrange(0, 360))
        self.surface = surface
        self.energy = energy
        self.variability_tax = variability_tax
        self.networks_entries = 12
        self.networks_outputs = 4
        self.train_turns = train_turns
        self.age = 0
        self.signals_to_receive = {}
        self.bots_list = bots_list
        self.bots_list.append(self)
        self.grass = grass
        self.surface = surface
        self.sound_list = []
        self.antennas_angles = [radians(30), radians(30), radians(30)]
        self.camuflage = 0
        self.last_couple = []
        self.size = 10
        self.in_ain = False
        self.relatives = []
        self.last_sex = 0
        self.center = sum(self.antennas_angles) / 2.0 % (2 * pi)
        self.max_wh = max(self.surface.get_size())
        if genetic_code == None:
            self.create_random_gcode()
        else:
            self.genetic_code = genetic_code
        self.neural_network = create_network(self.networks_entries,
                                             self.networks_outputs)
        self.get_gcode_information()

    def __add__(self, other):
        """Verify if the 2 bots have compatible gens and eat the
        same thing -> merge the 2 gens using crossing over tech.
        Return if the reproduction was realised"""
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        gen_compatibility = self.genetic_code ^ other.genetic_code
        gen_compatibility = bin(gen_compatibility)
        compatibility = 0
        for i in gen_compatibility[3:]:
            if int(i) == 0:
                compatibility += 1
        if compatibility > 50:
            return False
        cross_over = randint(0, 768)
        self_part = (self.genetic_code & ~(2 ** cross_over - 1))
        other_part = (other.genetic_code & (2 **  cross_over - 1))
        new_genome = self_part | other_part
        for i in range(int(self.variability_tax)):
            gen_to_chage = randint(0, 768)
            if choice([0,1]):
                new_genome |= (1 << gen_to_chage)
            else:
                new_genome &= ~(1 << gen_to_chage)
        son_energy = self.energy * 0.25 + other.energy *  0.25
        self.energy *= 0.75
        other.energy *= 0.75
        son_vt = (self.variability_tax + other.variability_tax) / 2.0
        sz = self.surface.get_size()
        self.__class__(randint(0,sz[0]), randint(0,sz[0]), self.surface,
                       self.bots_list, self.grass, son_energy, son_vt,
                       new_genome, self.train_turns)
        self.relatives.append(self.bots_list[-1])
        self.bots_list[-1].relatives.append(self)
        return True

    def train(self):
        #error_list = []
        for i in range(self.train_turns):
            tain_size = self.networks_entries * self.networks_outputs
            for to_train in range(tain_size):
                self.neural_network.train(self.trainset[0][to_train],
                                          self.trainset[1][to_train])
                #error = self.neural_network.get_MSE()
                #error_list.append(error)
        #plot(error_list)

    def create_random_gcode(self):
        """0 ~ 39 bits of genetic code = especie
        40 ~ 767 bits of genetic code = trainset"""
        self.genetic_code = 0
        for i in range(768):
            self.genetic_code += choice([0,1]) << i

    def get_gcode_information(self):
        """Read the genetic code
        Read the inputs of trainset in genetic code"""
        self.trainset = [[],[]]
        self.especie = self.genetic_code & (2 ** 40 - 1)
        gen = [0, 0]
        for i in range(48):
                self.trainset[0].append([])
                self.trainset[1].append([])
                for j in range(12):
                    if self.genetic_code & (1 << (gen[0] + 40)):
                        self.trainset[0][-1].append(1)
                    else:
                        self.trainset[0][-1].append(0)
                    gen[0] += 1
                for j in range(4):
                    if self.genetic_code & (1 << (gen[1] + 615)):
                        self.trainset[1][-1].append(1)
                    else:
                        self.trainset[1][-1].append(0)
                    gen[1] += 1
        self.train()

    def die(self):
        for rel in self.relatives:
            if self in rel.relatives:
                rel.relatives.remove(self)
            for rel2 in self.relatives:
                if rel != rel2 and not rel2 in rel.relatives:
                    rel.relatives.append(rel2)
        self.bots_list.remove(self)

    def move(self, well1, well2):
        #Move the bot depending of the well's activations
        #Enters values between 0 ~ 1
        self.angle += radians(well2 - well1) * 5
        self.angle %= (2 * pi)
        move = (1 - abs(well1 - well2)) * (well1 + well2) * 5
        self.pos[0] += cos(self.angle) * move
        self.pos[1] += sin(self.angle) * move
        size = self.surface.get_size()
        self.pos[0] %= size[0]
        self.pos[1] %= size[1]

    def emit_sound(self):
        self.sound_list.append([self.age, self.pos])
        for bot in self.bots_list:
            distance = bots_distance(bot, self)
            if not bot.age + int(distance / 50.0) in bot.signals_to_receive:
                bot.signals_to_receive[bot.age + int(distance / 50.0)] = []
            bot.signals_to_receive[bot.age + int(distance / 50.0)].append(
                                                           ["s",distance])

    def use_sensors(self, bots_by_distance = 0):
        sensors = [0.0] * 12
        aangles = self.antennas_angles[:]
        bots_by_distance = sorted(self.bots_list,      #remove itself from list
                                 key = lambda bot:bots_distance(bot, self))[1:]
        #angle base to verify the antennas
        astart = (self.angle - self.center) % (2 * pi)
        #Verify the 5 nearsts bots
        for bot in bots_by_distance[:5]:
            #angle between it self and other bot
            angle = atan2(bot.pos[1] - self.pos[1], bot.pos[0] - self.pos[0])
            angle %= (2 * pi)
            if angle > astart and angle <= (astart + aangles[0]) % (2 * pi):
                #Verify if the bot is of the same type of itself
                if bot.__class__.__name__ == self.__class__.__name__:
                    sensors[2] += bots_distance(bot, self) /self.max_wh
                else:
                    sensors[5] += bots_distance(bot, self) /self.max_wh
                sensors[8] += bot.camuflage
                continue
            astart = (astart + aangles[1]) % ( 2 * pi)
            if angle > astart and angle < (astart + aangles[1]) % (2 * pi):
                if bot.__class__.__name__ == self.__class__.__name__:
                    sensors[1] += bots_distance(bot, self) /self.max_wh
                else:
                    sensors[4] += bots_distance(bot, self) /self.max_wh
                sensors[7] += bot.camuflage
                continue
            astart = (astart + aangles[2]) % ( 2 * pi)
            if angle > astart and angle < (astart + aangles[2]) % (2 * pi):
                if bot.__class__.__name__ == self.__class__.__name__:
                    sensors[0] += bots_distance(bot, self) /self.max_wh
                else:
                    sensors[3] += bots_distance(bot, self) /self.max_wh
                sensors[6] += bot.camuflage
            astart = (self.angle - self.center) % (2 * pi)
        for sen in range(8):
            sensors[sen] /= 5.0
        sensors[9] = self.grass.get_food_quantity(self.pos[0], self.pos[1])
        if self.age in self.signals_to_receive:
            sig = self.signals_to_receive[self.age]
            for received in sig:
                if received[0] == "s":
                    sensors[10] = received[1]/self.max_wh

            del self.signals_to_receive[self.age]
        sensors[11] = (2000 - self.energy)/2000.0
        return sensors
                
class  Carnivore(Bot):


    def __init__(self, x, y, surface, bots_list, grass, energy = 500,
                 variability_tax = 1, genetic_code = None, train_turns = 100):
        Bot.__init__(self,x, y, surface, bots_list, grass, energy,
                     variability_tax, genetic_code, train_turns)

    def react(self):
        for bot in self.bots_list:
            if bot != self and bots_distance(bot,self) <(self.size + bot.size):
                sex_time = self.age - self.last_sex
                if sex_time > 100 and (self + bot):
                    self.last_sex = self.age
                    self.last_couple = [self, bot]
                    bot.last_couple = [bot, self]
                elif self.last_couple != [self, bot]:
                    if bot.__class__.__name__ == "Carnivore":
                        if bot.energy < self.energy:
                            self.energy += 400
                            bot.die()
                    else:
                        atack_force = self.energy / 2.0
                        retired_energy = atack_force / 4.0
                        bot.energy -= atack_force / 4.0
                        self.energy += atack_force / 4.0
                        if bot.energy <= 0:
                            self.energy += 300
                            bot.die()
        self.sensors = self.use_sensors()
        reactions = self.neural_network.run(self.sensors)
        self.camuflage = reactions[3] * 128
        self.move(reactions[0], reactions[1])
        if reactions[2] > 0.5:
            self.emit_sound()
        self.size = self.energy / 300.0  + 5 + self.age / 1000.0
        self.energy -= sum(reactions[0:3]) + (self.size - 5)
        if self.energy <= 0 or self.age > 5000:
            self.die()
        if self.size > 15:
            self.size = 15
        self.age += 1
        self.draw()

    def draw(self):
        if self.size < 2:
            self.size = 2
        #Body
        color = (255, 128 - self.camuflage, 128 - self.camuflage)
        pygame.draw.circle(self.surface, color, (int(self.pos[0]),
                           int(self.pos[1])), int(self.size))
        #Contour
        pygame.draw.circle(self.surface, (0,0,0), (int(self.pos[0]),
                           int(self.pos[1])), int(self.size), 1)
        #Antennas
        #-Antenna's points angles
        points = [(self.angle - self.center + self.antennas_angles[0] / 2) % (2 * pi)]
        for i in range(2):
            points.append((points[-1] + self.antennas_angles[i + 1])% (2 * pi))
        #-Antenna's colors
        colors = []
        for cor in range(3):
            r = self.sensors[cor] * 255 % 256
            g = self.sensors[3 + cor] * 255 % 256
            b = self.sensors[6 + cor] * 255 % 256
            colors.append((r, g, b))
        #-Draw then
        for ante in range(3):
            x = self.pos[0] + cos(points[ante]) * (self.size + 20)
            y = self.pos[1] + sin(points[ante]) * (self.size + 20)
            pygame.draw.aaline(self.surface, colors[ante], self.pos, (x,y))
        if self.in_ain:
            pygame.draw.circle(self.surface, (255,0,0), (int(self.pos[0]),
                               int(self.pos[1])), int(self.size + 3) , 1)
            #Sound Waves
            for sound in self.sound_list:
                size = int(self.age - sound[0]) * 50

                if size > max(self.surface.get_size()):
                    self.sound_list.remove(sound)
                    continue
                elif size > 2:
                    pygame.draw.circle(self.surface, (255,200,255),
                                      (int(sound[1][0]),int(sound[1][1])),
                                       size , 1)
            
            draw_genealogic_tree(self.surface, self)

class Herbivore(Bot):


    def __init__(self, x, y, surface, bots_list, grass, energy = 500,
                 variability_tax = 1, genetic_code = None, train_turns = 100):
        Bot.__init__(self,x, y, surface, bots_list, grass, energy,
                     variability_tax, genetic_code, train_turns)

    def react(self):
        for bot in self.bots_list:
            if bot != self and bots_distance(bot, self) < self.size + bot.size:
                if self.age - self.last_sex > 200 and self + bot:
                    self.last_sex = self.age
        self.energy += self.grass.eat(self.pos[0], self.pos[1])
        self.sensors = self.use_sensors()
        reactions = self.neural_network.run(self.sensors)
        self.move(reactions[0], reactions[1])
        if reactions[2] > 0.5:
            self.emit_sound()
        self.camuflage = reactions[3] * 128
        self.size = self.energy / 300.0  + 5 + self.age / 1000.0
        self.energy -= sum(reactions[0:3]) + (self.size - 5)
        if self.energy <= 0:
            self.die()
        elif self.energy > 2000 or self.age > 5000:
            self.energy = 2000
        if self.size > 15:
            self.size = 15
        self.age += 1
        self.draw()

    def draw(self):
        if self.size < 2:
            self.size = 2
        #Body
        color = (128 - self.camuflage, 128 - self.camuflage, 255)
        pygame.draw.circle(self.surface, color, (int(self.pos[0]),
                           int(self.pos[1])), int(self.size))
        #Contourbot
        pygame.draw.circle(self.surface, (0,0,0), (int(self.pos[0]),
                           int(self.pos[1])), int(self.size), 1)
        #Antennas
        #-Antenna's points angles
        points = [(self.angle - self.center + self.antennas_angles[0] / 2) % (2 * pi)]
        for i in range(2):
            points.append((points[-1] + self.antennas_angles[i + 1])% (2 * pi))
        #-Antenna's colors
        colors = []
        for cor in range(3):
            r = self.sensors[cor] * 255 % 256
            g = self.sensors[3 + cor] * 255 % 256
            b = self.sensors[6 + cor] * 255 % 256
            colors.append((r, g, b))
        #-Draw then
        for ante in range(3):
            x = self.pos[0] + cos(points[ante]) * (self.size + 20)
            y = self.pos[1] + sin(points[ante]) * (self.size + 20)
            pygame.draw.aaline(self.surface, colors[ante], self.pos, (x,y))
        if self.in_ain:
            pygame.draw.circle(self.surface, (255,0,0), (int(self.pos[0]),
                               int(self.pos[1])), int(self.size + 3), 1)
            #Sound Waves
            for sound in self.sound_list:
                size = int(self.age - sound[0]) * 50
                if size > max(self.surface.get_size()):
                    self.sound_list.remove(sound)
                    continue
                elif size > 2:
                    pygame.draw.circle(self.surface, (255,200,255),
                                      (int(sound[1][0]),int(sound[1][1])),
                                       size , 1)
            draw_genealogic_tree(self.surface, self)
        

class World:
    def __init__(self, size = (1000, 1000), grass_number = 30, min_herb = 4,
                 min_carn = 0, init_herb = 50, init_carn = 0):
        self.bots = []
        self.min_carn = min_carn
        self.min_herb = min_herb
        self.surface = pygame.display.set_mode(size)
        self.size = size
        self.running = True
        self.population_log = [[],[]]
        self.events = Thread(target=self.get_event)
        self.loop = Thread(target=self.main_loop)
        self.grass = Grass(grass_number, grass_number,
                           self.surface, initial_size = 50)
        if init_carn > 0:
            for families in range(2):
                new = Carnivore(randint(0,size[0]), randint(0,size[1]), self.surface,
                                self.bots, self.grass)
                for i in range((init_carn - 1) / 2):
                    new + new
                    self.bots[-1].energy = 500
        if init_herb > 0:
            for families in range(2):
                new = Herbivore(randint(0,size[0]), randint(0,size[1]), self.surface,
                                self.bots, self.grass)
                for i in range((init_herb - 1) / 2):
                    new + new
                    self.bots[-1].energy = 500

    def get_event(self):
        last_in_ain = self.bots[0]
        while 1:
            event = pygame.event.wait()
            if event.type == QUIT:
                self.running = False
                plot(self.population_log[0])
                plot(self.population_log[1])
                show()
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                #Bots by distance
                bbd = sorted(self.bots, key = lambda bot: bots_distance(bot,event))
                nearst_bot = bbd[0]
                if event.button == 1:
                    last_in_ain.in_ain = False
                    if bots_distance(nearst_bot,event) < nearst_bot.size:
                        nearst_bot.in_ain = True
                        last_in_ain = nearst_bot
                if event.button == 3:
                    #Bots by adaptation
                    bba = sorted(self.bots, key = lambda bot: bot.age)
                    best = bba[-1]
                    best.in_ain = True
                    last_in_ain.in_ain = False
                    last_in_ain = best
            if event.type == KEYDOWN:
                if event.unicode == "s":
                    print last_in_ain.genetic_code
                

    def population_controll(self):
        herb_pop = 0
        for bot in self.bots:
            if bot.__class__.__name__ == "Herbivore":
                herb_pop += 1
        carn_pop = len(self.bots) - herb_pop
        if herb_pop < self.min_herb:
            new_herb = Herbivore(randint(0,self.size[0]), randint(0,self.size[1]),
                                 self.surface, self.bots, self.grass)
            for i in range(self.min_herb):
                new_herb + new_herb
                self.bots[-1].energy = 500
        if carn_pop < self.min_carn:
            new_carn = Carnivore(randint(0,self.size[0]), randint(0,self.size[1]),
                                 self.surface, self.bots, self.grass)
            for i in range(self.min_carn):
                new_carn + new_carn
                self.bots[-1].energy = 500
        if (self.turn + 1) % 500 == 0:
            new_herb = Herbivore(randint(0,self.size[0]), randint(0,self.size[1]),
                                 self.surface, self.bots, self.grass)
            for i in range(5):
                new_herb + new_herb
                self.bots[-1].energy = 500
            new_carn = Carnivore(randint(0,self.size[0]), randint(0,self.size[1]),
                                 self.surface, self.bots, self.grass)
            for i in range(5):
                new_carn + new_carn
                self.bots[-1].energy = 500
        self.population_log[0].append(herb_pop)
        self.population_log[1].append(carn_pop)

    def main_loop(self):
        self.turn = 0
        while(self.running):
            self.population_controll()
            if self.turn % 1 == 0:
                pygame.display.flip()
                self.grass.draw()
            for bot in self.bots:
                bot.react()
            self.grass.grow()
            self.turn += 1

    def run(self):
        self.loop.start()
        self.events.start()

if __name__ == "__main__":
    simulation = World(size = (1000, 1000), grass_number = 30, min_herb = 0,
                       min_carn = 0, init_herb = 20, init_carn = 20)
    simulation.run()
