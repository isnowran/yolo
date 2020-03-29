#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:28:17 2019

@author: binary
"""

import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import requests
import collections
import cPickle

py = nn.Parameter(torch.tensor(0.1))
x = torch.tensor(0.5)
y = torch.tensor(0.5)

mse = nn.MSELoss()

def main2():
    for i in range(3):
        gx, gy = 1, 2
        print i
        print '-'*10
        out_x, out_y = px*x, py*y
        out = gx * out_x + gy * out_y
        loss = mse(target, out)
        print 'out', out.detach(), 'target', target.detach(), 'loss', loss.detach()
        
        optim.zero_grad()
        print loss, loss.grad
        loss1 = torch.tensor([loss])
        print loss1, loss1.grad
        
        loss1.backward()
        #loss.backward()
        optim.step()
        
        #print 'x:%s, y:%s, target:%s' % (x.detach(), y.detach(), target.detach())
        print 'px:%s, py:%s' % (px.detach(), py.detach())
        print
    
def f(a, y):
    return (a - y)**2

def df2(a, y, g):
    return 2*(a*g-y)*g

def df(a, y, g):
    return 2*g*(a-y)

def check(g):
    #px = nn.Parameter(torch.tensor(0.1))
    eta = 0.1
    px = torch.tensor(0.1, requires_grad=True)
    target = torch.tensor(1.0)

    optim = torch.optim.SGD([px], lr=eta)

    lose = g*f(px, target)
    lose.backward()
    print 'lose', lose
    print 'df', df(px, target, g)
    print 'px.grad', px.grad
    print 'self.step', px - px.grad * eta
    
    optim.step()
    print 'optim.step', px


def main():
    with open('./searchresults.html') as fd:
        buf = fd.read()
        soup = BeautifulSoup(buf, 'html5lib')
        
    rs = soup.find_all(class_='gridView')[0]
    rr = rs.find_all('ul')[0]

    savepath = './c.dump'
    try:
        with open(savepath) as fd:
            cdump = cPickle.load(fd)
    except Exception as e:
        print e, 'create new'
        cdump = collections.OrderedDict()
    else:
        print 'load dump', len(cdump)
        for url, it in cdump.iteritems():
            print '='*50
            print url
            for k, v in it.items():
                print '*', k
                print v if k != 'imgs_buf' else [len(_v) for _v in v]
                print
    
    for i, it in enumerate(rr):
        if type(it) != bs4.element.Tag:
            continue
        
        # out info
        link = it.find_all('a')[0].attrs['href'].strip()
        if link in cdump:
            print 'pass', link
            continue
        
        title = it.find_all(class_='detailscontainer')[0].text.strip()
        img0 = it.find_all('img')[0].attrs['src'].strip()
        price = it.find_all(class_='price')[0].text.strip()
    
        # info
        info = it.find_all(class_='infoContainer')[0].text
        info = ' '.join(info.split())
    
        print '# proc:', link
        r = requests.get(link)
        s2 = BeautifulSoup(r.text)
        p_real = s2.find(id="main_center_0_lblPriceRealizedPrimary").text.strip()
        p_estimate = s2.find(id="main_center_0_lblPriceEstimatedPrimary").text.strip()
        
        desc = s2.find(class_='lotDescription').text.strip()
        
        source = s2.find(class_='lot-details--provenance').text.strip()
        
        imgs_tag = s2.find_all(class_='cta-image carousel-slider-thumbnails--image')
        imgs = set([img.attrs['href'].strip() for img in imgs_tag])
        imgs_buf = []
        print 'imgs'
        for e, img in enumerate(imgs):
            print ' ', e, img
            r2 = requests.get(img)
            assert(r2.status_code == 200)
            imgs_buf.append(r2.content)
            '''
            with open('./tmp.jpg', 'w+b') as fd:
                fd.write(r2.content)
    
            with open('./tmp.jpg') as fd:
                img = Image.open(fd)
                img.show()
            '''
    
        print 'title:', title
        print 'price', price
        #print 'out_imgs:', img0
        print 'info', info
        print
        print 'price_estimate', p_estimate
        print 'price_real', p_real
        print
        print 'desc', desc
        print
        print source
        
        k = ['title', 'link', 'img0', 'price', 'info', \
             'p_real', 'p_estimate', 'desc', 'source', 'imgs', 'imgs_buf']
        
        v = [title, link, img0, price, info, \
             p_real, p_estimate, desc, source, imgs, imgs_buf]
        
        save = dict(zip(k, v))
        cdump[link] = save
    
    with open(savepath, 'w+b') as fd:
        cPickle.dump(cdump, fd)
        
main()
