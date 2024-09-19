#!/usr/bin/env waf

TOP = '.'
APPNAME = 'spng'

from waflib.extras import wcb
wcb.package_descriptions.insert(0, ("WCT", dict(
    incs=["WireCellUtil/Units.h"],
    libs=["WireCellUtil"], mandatory=True)))

def options(opt):
    opt.load('wcb')

def configure(cfg):
    cfg.load('wcb')

def build(bld):
    bld.load('smplpkgs')
    bld.smplpkg('WireCellSpng', use='WCT')


# def configure(cfg):
#     cfg.load('compiler_cxx')
#     cfg.load('smplpkgs')
#     do_configure(cfg, **package_descriptions)


# def build(bld):
#     pass


