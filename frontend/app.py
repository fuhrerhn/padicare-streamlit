"""PadiCare AI — Streamlit interface for cloud and local usage.

By default, the uploaded model is loaded directly for inference so Streamlit
Community Cloud needs no separate API service. The FastAPI integration remains
available: set PADICARE_INFERENCE_MODE=api and API_URL to use it locally.

Cloud demo history is stored only for the current browser session. For durable,
multi-user history, deploy the existing FastAPI backend with a persistent DB.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
from html import escape
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from collections import Counter
import logging
import sys

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.knowledge import get_advice
from backend.constants import CLASSES

log = logging.getLogger(__name__)

st.set_page_config(page_title="PadiCare AI | Kesehatan Tanaman Padi", page_icon="🌾",
                   layout="wide", initial_sidebar_state="collapsed")

# Image option: add frontend/assets/rice-field.jpg to avoid relying on the internet.
# The default is an Unsplash photograph by Ivan Bandura, licensed for free use.
DEFAULT_HERO_IMAGE = (
    "https://images.unsplash.com/photo-1646928998297-5fb2a10aca27"
    "?auto=format&fit=crop&w=2200&q=82"
)


def hero_image_url() -> str:
    """Prefer a project image and fall back to a remote landscape photograph."""
    assets = Path(__file__).resolve().parent / "assets"
    for name in ("rice-field.jpg", "rice-field.jpeg", "rice-field.webp", "rice-field.png"):
        path = assets / name
        if path.is_file():
            media = "image/png" if path.suffix == ".png" else (
                "image/webp" if path.suffix == ".webp" else "image/jpeg"
            )
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{media};base64,{encoded}"
    return DEFAULT_HERO_IMAGE


HERO_URL = hero_image_url()

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --pine: #193c2c;
  --green: #417a4b;
  --sage: #e8f1e7;
  --paper: #fffefa;
  --ink: #263b31;
  --body: #52685b;
  --hairline: #dce7da;
  --font-display: 'Plus Jakarta Sans', sans-serif;
  --font-body: 'DM Sans', sans-serif;
}
*, *::before, *::after {box-sizing: border-box;}
html, body, .stApp {font-family: var(--font-body); color: var(--ink);}
.stApp {background: #e8f1eb; overflow-x: clip;}
/* Native fallback if the navigation script is blocked or still loading. */
html, body, [data-testid='stAppViewContainer'], [data-testid='stMain'], section.main {
  scroll-behavior: smooth;
}
[data-testid='stAppViewContainer'] {overflow-x: clip;}
[data-testid='stHeader'] {background: transparent !important; height: 2rem;}
[data-testid='stToolbar'] {right: 1rem;}
.block-container {
  max-width: 1370px !important;
  padding: 0 24px 24px !important;
}
h1, h2, h3, h4 {font-family: var(--font-display) !important; letter-spacing: -.038em !important;}
[data-testid='stMarkdownContainer'] p {line-height: 1.65;}
button, input, textarea, select {font-family: var(--font-body);}
a:focus-visible, button:focus-visible, input:focus-visible {outline: 3px solid #f2d37c !important; outline-offset: 3px;}

/* A persistent navbar. It starts transparent over the field, then gains a surface. */
div[data-testid='stHorizontalBlock']:has(.site-nav) {
  position: fixed !important;
  top: 42px;
  left: 50%;
  transform: translateX(-50%);
  width: min(1300px, calc(100vw - 80px));
  height: 82px;
  min-height: 82px;
  z-index: 999;
  /* At the top: plain links directly on the rice-field photograph. */
  border: 1px solid transparent;
  border-radius: 100px;
  background: transparent;
  backdrop-filter: none;
  -webkit-backdrop-filter: none;
  box-shadow: none;
  padding: 0 23px;
  align-items: center !important;
  transition: background .35s, border-color .35s, box-shadow .35s, top .35s;
}
/* Streamlit nests columns in vertical blocks; centre those wrappers as well. */
div[data-testid='stHorizontalBlock']:has(.site-nav) > div[data-testid='stColumn'] {
  align-self: stretch !important;
  display: flex !important;
  flex-direction: column;
  justify-content: center !important;
  min-height: 0 !important;
}
div[data-testid='stHorizontalBlock']:has(.site-nav) > div[data-testid='stColumn'] > div[data-testid='stVerticalBlock'] {
  justify-content: center !important;
  gap: 0 !important;
  min-height: 0 !important;
}
div[data-testid='stHorizontalBlock']:has(.site-nav) .site-nav {
  height: 56px;
  min-height: 0;
}
div[data-testid='stHorizontalBlock']:has(.site-nav) [data-testid='stMarkdownContainer'] > p {
  margin: 0;
}
div[data-testid='stHorizontalBlock']:has(.site-nav).nav-scrolled {
  top: 12px;
  /* After scrolling: compact pill with a visible surface and soft shadow. */
  background: rgba(255, 254, 250, .97);
  border-color: rgba(218, 229, 217, .95);
  box-shadow: 0 11px 34px rgba(20, 47, 30, .12);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
}
.site-nav {display: flex; justify-content: space-between; align-items: center; min-height: 51px; gap: 20px;}
.site-brand {display: inline-flex; align-items: center; gap: 11px; text-decoration: none !important; white-space: nowrap;}
.site-symbol {width: 42px; height: 42px; flex: 0 0 42px; border-radius: 14px; background: #e4eccb; display: grid; place-items: center;}
.site-symbol svg {width: 24px; height: 24px;}
.site-name {display:block; font: 800 1.03rem/1.35 var(--font-display); color: #fff; letter-spacing: -.04em;}
.site-tag {display:block; color: #e1e9de; font-size: .69rem; line-height: 1.55;}
.site-links {display:flex; justify-content:flex-end; align-items:center; gap: clamp(10px, 1.7vw, 29px);}
.site-links a {color: #f4f8f1 !important; text-decoration: none !important; font-weight: 650; font-size: .87rem; padding: 9px 0; white-space: nowrap; transition: opacity .2s, color .2s;}
.site-links a:hover {opacity: .76;}
.site-links .menu-cta {color: #f4f8f1 !important; background: transparent; border-radius: 50px; padding: 12px 19px; font-weight: 800;}
.nav-scrolled .site-name, .nav-scrolled .site-links a:not(.menu-cta) {color: var(--pine) !important;}
.nav-scrolled .site-tag {color: #657f6b;}
.nav-scrolled .site-symbol {background: #4d8a60;}
.nav-scrolled .site-symbol path {stroke: white;}
.nav-scrolled .site-links .menu-cta {background: #447b52; color: #fff !important;}

/* Compact language switch. Keep Streamlit's native toggle for state and keyboard access. */
.st-key-language_toggle {
  --switch-width: 52px;
  position: relative;
  /* Native Streamlit toggle wrapper sits above the navigation text baseline.
     Offset only this control so ID, track and EN share the menu centerline. */
  top: 9px;
  display: flex !important;
  align-items: center;
  justify-content: center;
  width: 116px;
  min-width: 116px;
  height: 36px;
  margin: 0 !important;
}
.st-key-language_toggle::before,
.st-key-language_toggle::after {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  z-index: 2;
  font: 700 .76rem/1 var(--font-display);
  letter-spacing: .01em;
  color: #f3f6ed;
  pointer-events: none;
}
.st-key-language_toggle::before {content: 'ID'; left: 0;}
.st-key-language_toggle::after {content: 'EN'; right: 0;}
.st-key-language_toggle [data-testid='stCheckbox'] {
  position: absolute;
  left: 32px;
  top: 50%;
  transform: translateY(-50%);
  width: var(--switch-width);
  height: 28px;
  min-height: 28px;
  background: #e8f1e7;
  border: 1px solid #b8d0b9;
  border-radius: 100px;
  box-shadow: inset 0 1px 2px rgba(27, 67, 38, .06);
  transition: background .2s, border-color .2s;
}
.st-key-language_toggle [data-testid='stCheckbox']::after {
  content: '';
  position: absolute;
  top: 3px;
  left: 4px;
  width: 20px;
  height: 20px;
  background: #437d50;
  border-radius: 50%;
  box-shadow: 0 1px 3px rgba(22, 63, 36, .18);
  pointer-events: none;
  transition: transform .22s ease;
}
.st-key-language_toggle:has(input:checked) [data-testid='stCheckbox'] {
  background: #dcecdf;
  border-color: #a5c8ab;
}
.st-key-language_toggle:has(input:checked) [data-testid='stCheckbox']::after {
  transform: translateX(24px);
}
/* Expand the invisible native label across ID, switch, and EN for easy clicking. */
.st-key-language_toggle [data-testid='stCheckbox'] label {
  position: absolute !important;
  top: -4px;
  left: -32px;
  width: 116px !important;
  height: 36px !important;
  opacity: 0 !important;
  cursor: pointer !important;
  z-index: 5;
}
.st-key-language_toggle input {accent-color: #437d50;}
.st-key-language_toggle:focus-within [data-testid='stCheckbox'] {outline: none;}
.st-key-language_toggle:has(input:focus-visible) [data-testid='stCheckbox'] {
  outline: 2px solid #4d9664;
  outline-offset: 3px;
}
.nav-scrolled .st-key-language_toggle::before,
.nav-scrolled .st-key-language_toggle::after {color: var(--pine);}
.nav-scrolled .st-key-language_toggle [data-testid='stCheckbox'] {border-color: #bed2c0;}

/* Full-bleed hero; no card/radius and no white frame around the photo. */
#beranda, #tentang, #deteksi, #panduan, #riwayat {scroll-margin-top: 110px;}
.hero-photo {
  width: 100vw;
  min-height: clamp(590px, 83vh, 830px);
  position: relative;
  left: 50%;
  margin-left: -50vw;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: center;
  isolation: isolate;
  padding: 158px max(6.5vw, calc((100vw - 1250px) / 2)) 95px;
  background-image: linear-gradient(90deg, rgba(8,34,22,.80) 0%, rgba(8,34,22,.62) 37%, rgba(8,34,22,.11) 100%), url('__HERO_URL__');
  background-size: cover;
  background-position: center 57%;
  background-color: #21432a;
}
.hero-photo::before {content: ''; position:absolute; inset:0; z-index:-1; background:linear-gradient(0deg,rgba(12,35,18,.38),transparent 38%);}
.hero-content {max-width: 800px; animation: rise-in .8s both;}
.hero-kicker {font: 600 .81rem/1.4 var(--font-body); color: #e9f0d3; letter-spacing: .035em;}
.hero-kicker::before {content:''; display:inline-block; width: 24px; height: 2px; margin:0 11px 4px 0; background:#dee8bb;}
.hero-photo h1 {color: #fff!important; font: 750 clamp(3.1rem,5.3vw,5.6rem)/1.11 var(--font-display); letter-spacing:-.066em!important; margin:25px 0 18px !important; max-width:790px; text-wrap:balance;}
.hero-photo h1 span {color:#e2efb2;}
.hero-photo p {color: #f3f6eb; font-size: 1.12rem; line-height:1.78; max-width:540px; margin: 0 0 32px;}
.hero-cta {display:inline-flex;align-items:center;gap:20px;background:#e9efd1;color:#163d2b!important;border-radius:50px;text-decoration:none!important;font-weight:800;padding:15px 23px;transition:transform .25s, background .25s;}
.hero-cta:hover {background:#fff; transform:translateY(-3px);}
.hero-caption {position:absolute;bottom:26px;left:max(6.5vw,calc((100vw - 1250px)/2));font-size:.78rem;color:rgba(255,255,255,.79);}
@keyframes rise-in {from{opacity:.25;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}

/* A single, gap-free document: Streamlit normally inserts 1rem between blocks.
   Scope this override to the ONE top-level stack containing the hero and scanner,
   so nested forms, columns, and the recommendation accordions retain their spacing. */
.block-container > [data-testid='stVerticalBlock'],
.block-container > [data-testid='stVerticalBlockBorderWrapper'] > [data-testid='stVerticalBlock'],
[data-testid='stVerticalBlock']:has(.hero-photo):has(.st-key-deteksi) {
  gap: 0 !important;
  row-gap: 0 !important;
}
[data-testid='stVerticalBlock']:has(.hero-photo):has(.st-key-deteksi) > div {
  margin-block: 0 !important;
}
/* Anchor IDs live inside the heading markup instead of occupying a standalone
   Streamlit element (which previously created a blank band). */
.section-anchor {display:block; height:0; line-height:0; scroll-margin-top:112px;}

/* The colours meet at the edges: no independent white bands between sections. */
.proof-strip {
  position:relative; isolation:isolate;
  display:flex;gap:35px;align-items:center;flex-wrap:wrap;
  padding:24px 5px 22px;margin:0;border:0;
}
.proof-strip::before {
  content:'';position:absolute;z-index:-1;top:0;bottom:0;left:50%;
  transform:translateX(-50%);width:100vw;
  background:linear-gradient(180deg,#f5f8ef 0%,#f7f8f1 100%);
}
.proof-item {display:flex;align-items:baseline;gap:9px;color:#627568;font-size:.83rem;}
.proof-item b {font:800 1.17rem var(--font-display);color:#315d3d;}
.section-header {margin:0 0 26px;}
.section-header h2 {font-size:clamp(1.9rem,2.9vw,2.7rem);font-weight:750;color:var(--pine);line-height:1.25;margin:0 0 10px!important;}
.section-header p {max-width:650px;color:var(--body);margin:0;font-size:1rem;line-height:1.72;}
.section-eyebrow {font-size:.9rem;letter-spacing:0;color:#567b60;font-weight:650;margin:0 0 10px;text-transform:none;}
.editorial {padding:61px 2px 57px;display:grid;grid-template-columns: minmax(310px, .92fr) minmax(380px,1.08fr);gap: clamp(45px,7vw,105px);align-items:center;position:relative;isolation:isolate;}
.editorial::before {content:'';position:absolute;inset:0;left:50%;width:100vw;transform:translateX(-50%);z-index:-1;background:linear-gradient(180deg,#f7f8f1 0%,#f1f6ed 52%,#edf3eb 100%);}
.process {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;}
.process-item {position:relative;border-top:2px solid #9cb999;padding:18px 4px 0;}
.process-no {font:750 .87rem var(--font-display);color:#548862;}
.process-title {font:750 1.02rem/1.4 var(--font-display);color:#234932;margin:12px 0 7px;}
.process-item p {font-size:.88rem;color:#607666;line-height:1.65;margin:0;}

/* Section fills go all the way to the viewport edges, but content remains readable. */
.st-key-deteksi,.st-key-panduan,.st-key-saran,.st-key-riwayat {position:relative; isolation:isolate;background:transparent;border:0;border-radius:0;padding:63px 2px 66px;margin:0;}
.st-key-deteksi::before,.st-key-panduan::before,.st-key-saran::before,.st-key-riwayat::before {content:'';position:absolute;z-index:-1;top:0;bottom:0;left:50%;width:100vw;transform:translateX(-50%);}
.st-key-deteksi::before {
  background:
    radial-gradient(ellipse at 89% 22%,rgba(145,196,147,.25) 0%,transparent 48%),
    linear-gradient(180deg,#edf3eb 0%,#e5f0e8 52%,#e8f2eb 100%);
}
.st-key-saran::before {
  background:radial-gradient(ellipse at 15% 70%,rgba(239,213,145,.15),transparent 53%),
             linear-gradient(180deg,#e8f2eb 0%,#f0f3e7 55%,#f7f4e9 100%);
}
.st-key-panduan::before {
  background:radial-gradient(ellipse at 88% 32%,rgba(220,198,141,.17),transparent 48%),
             linear-gradient(180deg,#f7f4e9 0%,#fcf9f0 55%,#f5f2e9 100%);
}
.st-key-riwayat::before {
  background:radial-gradient(ellipse at 8% 77%,rgba(164,207,170,.25),transparent 48%),
             linear-gradient(180deg,#f5f2e9 0%,#e8f0e8 52%,#e8f1eb 100%);
}
/* Before a scan, the optional recommendation section does not exist.
   The guide then takes over its colour transition instead of jumping to cream. */
.stApp:not(:has(.st-key-saran)) .st-key-panduan::before {
  background:radial-gradient(ellipse at 88% 34%,rgba(220,198,141,.13),transparent 48%),
             linear-gradient(180deg,#e8f2eb 0%,#f7f4e9 52%,#f5f2e9 100%);
}
.scan-lead{display:flex;justify-content:space-between;align-items:end;gap:24px;margin-bottom:20px;}
.scan-side-note {max-width:320px;color:#687d6d;font-size:.88rem;line-height:1.6;text-align:right;}
.panel-kicker {font-size:.9rem;color:#3e6a4b;font-weight:700;margin-bottom:12px;}
.result-empty {min-height:200px;border:1px solid #d7e5d6;border-radius:17px;background:rgba(255,255,255,.83);padding:34px 32px;display:flex;flex-direction:column;justify-content:center;}
.result-empty h3 {font-size:1.2rem;font-weight:750;margin:0 0 7px!important;color:#254c35;}
.result-empty p {font-size:.92rem;line-height:1.7;color:#687c6b;max-width:460px;margin:0;}
.diagnosis-panel {background:#fff;border:1px solid #d6e7d7;padding:23px 26px;border-radius:17px;}
.diagnosis-kicker {font-size:.87rem;color:#5a7964;font-weight:650;}
.diagnosis-code {font:800 clamp(1.23rem,1.6vw,1.55rem)/1.38 var(--font-display);color:#1f5838;overflow-wrap:anywhere;margin:9px 0 6px;}
.diagnosis-explain {font-size:.87rem;color:#657d6b;margin-bottom:16px;}
.diagnosis-score {font:800 2.4rem var(--font-display);letter-spacing:-.055em;color:#33694a;}
.diagnosis-foot {color:#627968;font-size:.85rem;}
.prob-row{margin:14px 0;}
.prob-heading{display:flex;justify-content:space-between;gap:15px;font-size:.88rem;color:#294b35;font-weight:700;margin-bottom:8px;}
.prob-heading span:first-child {overflow-wrap:anywhere;}
.prob-track{height:7px;background:#dceada;border-radius:99px;overflow:hidden;}
.prob-fill{height:100%;border-radius:99px;background:#70a079;transform-origin:left;animation:growBar .8s both;}
@keyframes growBar {from{transform:scaleX(0)}to{transform:scaleX(1)}}
.knowledge-intro{margin:0 0 18px;padding:0 0 17px;border-bottom:1px solid #e1e7dc;}
.knowledge-intro h3{font:750 1.32rem var(--font-display);color:#264633;overflow-wrap:anywhere;margin:0 0 7px;}
.knowledge-intro p{color:#607364;line-height:1.75;margin:0;max-width:860px;}
.note-icon {flex:0 0 24px;width:24px;height:24px;border:1px solid #699477;border-radius:50%;display:grid;place-items:center;color:#3f7954;font:700 .86rem Georgia,serif;font-style:italic;}
.callout {display:flex;align-items:flex-start;gap:12px;max-width:880px;color:#42604d;padding:14px 17px;background:rgba(225,238,223,.7);border-left:3px solid #8eaf8b;border-radius:0 8px 8px 0;font-size:.88rem;line-height:1.7;margin:20px 0 16px;}
[data-testid='stFileUploader'] {background:rgba(255,255,255,.9);border:1.6px dashed #a7c9a8;border-radius:14px;padding:12px;}
[data-testid='stFileUploader'] section {background:transparent;}
[data-testid='stExpander'] {border-color:#dae5d8;border-radius:10px;background:rgba(255,255,255,.69);}
.stButton>button[kind='primary'],.stDownloadButton>button[kind='primary']{background:#367548;color:white;border:1px solid #367548;border-radius:10px;min-height:3rem;font-weight:750;transition:background .2s, transform .2s;}
.stButton>button[kind='primary']:hover,.stDownloadButton>button[kind='primary']:hover{background:#245a38;color:white;transform:translateY(-1px);}
.stButton>button[kind='secondary'],.stDownloadButton button{border-radius:10px;border-color:#c9d9cb;color:#2c6340;font-weight:650;}
[data-testid='stMetric'] {background:none;border:0;}
[data-testid='stMetricValue'] {font:750 1.5rem var(--font-display);color:#315d3d;}
.st-key-riwayat [data-testid='stMetric'] {border-bottom:1px solid rgba(74,120,88,.15);padding-bottom:14px;}
.footer {
  position:relative;isolation:isolate;display:flex;justify-content:space-between;
  flex-wrap:wrap;gap:18px;padding:29px 3px 33px;margin:0;
  color:#5e7967;font-size:.78rem;
}
.footer::before {
  content:'';position:absolute;inset:0;left:50%;width:100vw;
  transform:translateX(-50%);z-index:-1;
  background:linear-gradient(180deg,#e8f1eb 0%,#e4eee7 100%);
}
.footer a{color:#2a6e43!important;}
/* Optional scroll reveal, enabled only after the JS observer is ready.
   Content remains fully visible if scripts are blocked. */
html.paddy-motion .section-header.reveal-ready,
html.paddy-motion .editorial .process-item.reveal-ready {
  opacity:0;
  transform:translateY(22px);
  transition:opacity .62s ease,transform .75s cubic-bezier(.2,.7,.2,1);
}
html.paddy-motion .section-header.reveal-ready.in-view,
html.paddy-motion .editorial .process-item.reveal-ready.in-view {
  opacity:1;
  transform:translateY(0);
}
html.paddy-motion .editorial .process-item:nth-child(2){transition-delay:.09s;}
html.paddy-motion .editorial .process-item:nth-child(3){transition-delay:.18s;}
/* Quiet background motion makes the sections feel alive without competing with text. */
.st-key-deteksi::before,.st-key-saran::before,.st-key-panduan::before,.st-key-riwayat::before,
.editorial::before,.proof-strip::before,.footer::before {pointer-events:none;}
/* Animations are decorative, never required for reading the page. */
.process-item {transition:transform .24s, border-color .24s;}
.process-item:hover {transform:translateY(-3px);border-color:#458456;}

@media (max-width:1050px) {
  div[data-testid='stHorizontalBlock']:has(.site-nav){width:calc(100vw - 36px);padding:0 15px;}
  .site-links{gap:12px;}
  .site-links a{font-size:.78rem;}
  .editorial{gap:35px;grid-template-columns:1fr;}
}
@media (max-width:760px) {
  .block-container{padding:0 17px 20px!important;}
  div[data-testid='stHorizontalBlock']:has(.site-nav){top:17px;border-radius:23px;height:auto;min-height:78px;padding:8px 14px;}
  .site-nav{flex-wrap:wrap;}
  .site-links{justify-content:flex-start;flex-wrap:wrap;gap:8px 14px;}
  .site-links a{font-size:.75rem;padding:4px 0;}
  .site-links .menu-cta{padding:7px 12px;}
  .site-name{font-size:.94rem;}
  .site-symbol{width:35px;height:35px;flex-basis:35px;border-radius:11px;}
  .hero-photo{min-height:690px;padding:215px 25px 78px;background-position:58% center;}
  .hero-photo h1{font-size:clamp(2.55rem,9vw,3.85rem);}
  .hero-photo p{font-size:.99rem;}
  .hero-caption{left:25px;}
  .proof-strip{gap:15px 25px;}
  .process{grid-template-columns:1fr;gap:14px;}
  .process-item{padding:10px 0 4px;}
  .st-key-deteksi,.st-key-saran,.st-key-panduan,.st-key-riwayat{padding:46px 0 50px;}
  .scan-lead{display:block;}
  .scan-side-note{text-align:left;margin-top:10px;}
}
@media (max-width:460px) {
  div[data-testid='stHorizontalBlock']:has(.site-nav){width:calc(100vw - 20px);padding:8px 10px;}
  .st-key-language_toggle {transform:scale(.96);transform-origin:right center;}
  .hero-photo{padding-top:236px;min-height:700px;}
  .hero-photo h1{font-size:2.4rem;}
  .proof-item{font-size:.76rem;}
  .hero-caption{font-size:.71rem;}
}
@media (prefers-reduced-motion:reduce) {
  html,body,[data-testid='stAppViewContainer'],[data-testid='stMain'],section.main{scroll-behavior:auto!important;}
  *,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important;}
  html.paddy-motion .section-header.reveal-ready,
  html.paddy-motion .editorial .process-item.reveal-ready {
    opacity:1!important;transform:none!important;
  }
}
</style>
""".replace("__HERO_URL__", HERO_URL)
st.markdown(CSS, unsafe_allow_html=True)

TR = {
'id': {
 'tagline':'PEMERIKSAAN KESEHATAN PADI',
 'hero_title':'Kenali kondisi padi, mulai dari satu foto.',
 'hero_body':'Unggah foto tanaman untuk mengetahui kemungkinan kondisinya dan mendapatkan langkah pemeriksaan serta penanganan awal.',
 'hero_pill':'🌱 10 kelas kondisi  ·  ⚡ Analisis dalam hitungan detik  ·  🌍 Dua bahasa',
 'lang':'Bahasa / Language','menu':'Navigasi','tab_scan':'Deteksi','tab_history':'Riwayat',
 'tab_library':'Panduan Penyakit','tab_about':'Tentang',
 'how':'Analisis foto tanaman Anda','how_desc':'Ambil foto tajam, fokus pada gejala, dengan pencahayaan yang cukup. Untuk gejala di malai atau batang, sertakan bagian tanaman tersebut.',
 'input_method':'Metode input gambar','upload':'Unggah gambar','camera':'Ambil foto kamera','upload_label':'Tarik atau pilih foto daun/tanaman padi',
 'photo_label':'Ambil foto tanaman padi','photo_preview':'Pratinjau foto','analyze':'Analisis Foto',
 'wait':'Model sedang memproses foto...','api_ok':'API tersedia','api_no_model':'API aktif, tetapi file model belum tersedia',
 'api_down':'API belum terhubung','api_help':'Jalankan backend FastAPI dan pastikan file model .pth ada di models/.',
 'result':'Hasil analisis AI','top':'Top 3 kemungkinan model','conf':'Confidence model','prob':'Skor softmax model, bukan tingkat kepastian diagnosis',
 'uncertain':'Perlu verifikasi lebih lanjut','uncertain_desc':'Skor tertinggi rendah atau jarak antarkelas kecil. Ambil beberapa foto tambahan dan konsultasikan gejalanya.',
 'not_certain':'Confidence tinggi pun tidak menjamin diagnosis benar. Pertimbangkan seluruh gejala lapangan.',
 'signs':'Gejala yang perlu diperiksa','knowledge':'Mengenal kondisi ini','steps':'Rekomendasi tindakan bertahap',
 'step1':'Tahap 1 · Pemeriksaan awal','step2':'Tahap 2 · Pemantauan','step3':'Tahap 3 · Pencegahan',
 'sources':'Sumber rujukan','download':'Unduh ringkasan diagnosis (.txt)',
 'healthnote':'Panduan ini bersifat edukatif, bukan resep pestisida atau diagnosis definitif. Pilihan pestisida dan dosis hanya setelah konfirmasi petugas serta mengikuti label dan ketentuan setempat.',
 'history_title':'Riwayat & analitik deteksi','history_caption':'Hanya metadata prediksi yang disimpan ke SQLite. Foto tidak disimpan.',
 'total':'Total pemeriksaan','found':'Kelas yang pernah terdeteksi','latest':'Prediksi terakhir',
 'hist_list':'Pemeriksaan terbaru','hist_chart':'Distribusi hasil diagnosis','empty':'Belum ada data. Unggah foto di menu Deteksi Tanaman untuk memulai.',
 'refresh':'Muat ulang data','csv':'Unduh riwayat CSV','lib_title':'Ensiklopedia Kesehatan Padi',
 'lib_desc':'Pelajari gejala, tindak lanjut, dan pencegahan untuk seluruh 10 label yang digunakan model.',
 'pick':'Pilih kondisi','about_title':'Mengenal PadiCare AI','about_desc':'Sistem klasifikasi gambar berbasis EfficientNet-B0 yang dikembangkan sebagai proyek Data Science & Machine Learning.',
 'metric1':'Akurasi test set','metric2':'Macro F1 test set','metric3':'Jumlah kelas','model_note':'Angka evaluasi berasal dari notebook pengembangan (holdout test), bukan akurasi per foto atau validasi lapangan.',
 'data_note':'Arsitektur: EfficientNet-B0 (frozen feature extractor) + classifier head. Backend: FastAPI + SQLite. Frontend: Streamlit.',
 'errors':'Gagal menghubungi API','help_photo':'Format JPEG, PNG, atau WEBP, maksimum 8 MB.',
 'id':'ID prediksi','created':'Waktu (UTC)','class':'Kondisi','score':'Confidence',
 'disclaimer':'Keputusan lapangan tetap harus mempertimbangkan kondisi tanaman, musim, lingkungan, dan verifikasi petugas pertanian.',
 'privacy':'Privasi: aplikasi menyimpan hasil klasifikasi dan waktu pemeriksaan saja, bukan foto atau identitas petani.',
 'public_history':'Mode publik: riwayat individu disembunyikan; hanya agregat yang ditampilkan.',
 'warning_normal':'Label normal berarti foto tampak sehat menurut model, bukan jaminan tanaman bebas masalah.',
 'reference':'Referensi dan verifikasi agronomis',
},
'en': {
 'tagline':'RICE HEALTH SCREENING',
 'hero_title':'Understand your rice plant, one photo at a time.',
 'hero_body':'Upload a plant photo to review possible conditions, model confidence, and practical first steps.',
 'hero_pill':'🌱 10 health categories  ·  ⚡ Fast image analysis  ·  🌍 Bilingual',
 'lang':'Language / Bahasa','menu':'Navigation','tab_scan':'Scan Plant','tab_history':'History',
 'tab_library':'Disease Guide','tab_about':'About',
 'how':'Analyze your rice plant','how_desc':'Use a sharp, well-lit photo of symptoms. Include panicles or stems when signs occur there.',
 'input_method':'Image input method','upload':'Upload image','camera':'Use camera','upload_label':'Choose or drag a rice plant photo',
 'photo_label':'Capture a rice plant photo','photo_preview':'Image preview','analyze':'Analyze Photo',
 'wait':'Analyzing your photo...','api_ok':'API connected','api_no_model':'API online, but the model file is missing',
 'api_down':'Cannot connect to API','api_help':'Start FastAPI and place the model .pth file in models/.',
 'result':'AI analysis result','top':'Top 3 model predictions','conf':'Model confidence','prob':'Softmax model score, not diagnostic certainty',
 'uncertain':'Further verification needed','uncertain_desc':'Top confidence is low or predictions are close. Retake multiple photos and consult an expert.',
 'not_certain':'Even high confidence does not guarantee a correct diagnosis. Inspect the whole plant.',
 'signs':'Symptoms to inspect','knowledge':'About this condition','steps':'Step-by-step recommendations',
 'step1':'Step 1 · First checks','step2':'Step 2 · Monitoring','step3':'Step 3 · Prevention',
 'sources':'Information sources','download':'Download diagnosis summary (.txt)',
 'healthnote':'Educational advice only, not a pesticide prescription or definitive diagnosis. Use any pesticide only after local expert confirmation and in accordance with registered labels and regulations.',
 'history_title':'Diagnosis history & analytics','history_caption':'Only prediction metadata is stored in SQLite. Images are not retained.',
 'total':'Total screenings','found':'Categories observed','latest':'Latest prediction',
 'hist_list':'Recent screenings','hist_chart':'Classification distribution','empty':'No data yet. Start by analyzing a photo in the Plant Scanner tab.',
 'refresh':'Refresh data','csv':'Download history CSV','lib_title':'Rice Health Knowledge Hub',
 'lib_desc':'Explore symptoms, follow-up actions, and prevention across all 10 model categories.',
 'pick':'Select a condition','about_title':'About PadiCare AI','about_desc':'An EfficientNet-B0-based image classifier developed for a Data Science & Machine Learning final project.',
 'metric1':'Holdout test accuracy','metric2':'Holdout macro F1','metric3':'Classes','model_note':'Performance figures come from the development notebook holdout test, not per-image accuracy or field validation.',
 'data_note':'Architecture: EfficientNet-B0 (frozen feature extractor) + classifier head. Backend: FastAPI + SQLite. Frontend: Streamlit.',
 'errors':'API request failed','help_photo':'JPEG, PNG, or WEBP; maximum 8 MB.',
 'id':'Prediction ID','created':'Timestamp (UTC)','class':'Condition','score':'Confidence',
 'disclaimer':'Field decisions must consider symptoms, growing season, local conditions, and agricultural expert verification.',
 'privacy':'Privacy: only prediction metadata and timestamps are retained, not images or farmer identities.',
 'public_history':'Public mode: individual history is hidden; only aggregates are displayed.',
 'warning_normal':'The normal label means the photo appears healthy to the model, not that the entire crop is disease-free.',
 'reference':'References and agronomy verification',
},
}

def api_get(path: str, timeout: int = 8):
    """Read session-level demo data, or query FastAPI when API mode is enabled."""
    if DIRECT_INFERENCE:
        records = st.session_state.get("demo_history", [])
        if path == "/health":
            return {"status": "ok", "model_file_exists": MODEL_FILE.is_file()}
        if path.startswith("/history"):
            return records[:100]
        if path == "/stats":
            counts = Counter(row["label"] for row in records)
            return {"total": len(records),
                    "distribution": {name: counts.get(name, 0) for name in CLASSES}}
        raise ValueError("Unknown endpoint")
    response = requests.get(f"{BASE_URL}{path}", headers=AUTH_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_resource(show_spinner=False)
def load_model():
    """Keep one CPU model in memory between Streamlit reruns."""
    from backend.model import PaddyCarePredictor
    return PaddyCarePredictor(str(MODEL_FILE))


def predict_on_streamlit(image_bytes: bytes) -> dict:
    """Verify the uploaded photo and run the same inference as the FastAPI backend."""
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(image_bytes)) as probe:
        probe.verify()
    with Image.open(io.BytesIO(image_bytes)) as photo:
        if photo.width * photo.height > 24_000_000:
            raise ValueError("Image resolution is too large")
        rgb_image = ImageOps.exif_transpose(photo).convert("RGB")
    output = load_model().predict(rgb_image)
    result = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **output,
    }
    if "demo_history" not in st.session_state:
        st.session_state["demo_history"] = []
    st.session_state["demo_history"].insert(0, {
        key: result[key] for key in ("id", "created_at", "label", "confidence")
    })
    st.session_state["demo_history"] = st.session_state["demo_history"][:100]
    return result


def heading_list(items: list[str]):
    for item in items:
        st.markdown(f'- {item}')



def report_text(result: dict, lang: str, t: dict) -> str:
    label = result['label']
    a = get_advice(label, lang)
    lines = ['PadiCare AI — Rice Health Screening',
             f"Prediction ID: {result['id']}",
             f"Date (UTC): {result['created_at']}",
             f"Model label: {label}",
             f"{t['conf']}: {100*result['confidence']:.2f}%", '',
             f"{t['top']}:"]
    lines += [f"- {key}: {100*val:.2f}%"
              for key, val in sorted(result['probabilities'].items(), key=lambda it:-it[1])[:3]]
    lines.extend(['', f"{t['knowledge']}: {a['overview']}"])
    for section, key in ((t['signs'],'signs'),(t['step1'],'initial'),(t['step2'],'monitor'),(t['step3'],'prevent')):
        lines.append(f'\n{section}:')
        lines += [f'- {item}' for item in a[key]]
    lines.append('\n'+t['healthnote'])
    lines.append(t['disclaimer'])
    for source in a['sources']:
        lines.append(source['title']+': '+source['url'])
    return '\n'.join(lines)


# Override technical wording only in the visitor-facing interface.
TR['id'].update({
    'hero_body': 'Unggah foto padi, lihat kemungkinan masalahnya, lalu ikuti saran pemeriksaan awal.',
    'conf': 'Tingkat keyakinan AI',
    'prob': 'Angka ini bukan jaminan bahwa hasil pemeriksaan pasti benar.',
    'uncertain': 'Hasil perlu diperiksa kembali',
    'uncertain_desc': 'Hasil belum cukup meyakinkan. Coba foto ulang dengan lebih jelas atau tanyakan kepada penyuluh pertanian.',
    'not_certain': 'Tetap periksa kondisi tanaman secara langsung sebelum mengambil tindakan.',
    'healthnote': 'Saran ini hanya panduan awal. Periksa keadaan tanaman di sawah dan tanyakan kepada penyuluh pertanian sebelum menggunakan pestisida.',
    'lib_desc': 'Pilih kondisi tanaman untuk melihat tanda-tanda dan penanganan awal.',
    'history_title': 'Riwayat pemeriksaan',
    'history_caption': 'Lihat hasil pemeriksaan sebelumnya. Foto tidak disimpan.',
    'how_desc': 'Foto bagian tanaman yang bermasalah dengan cahaya cukup dan gambar yang jelas.',
    'top': 'Tiga kemungkinan hasil',
    'steps': 'Langkah penanganan awal',
    'signs': 'Ciri-ciri yang bisa diperiksa',
    'model_note': 'Angka ini berasal dari pengujian model pada data uji, bukan jaminan untuk setiap foto.',
})
TR['en'].update({
    'hero_body': 'Upload a rice plant photo, review a possible issue, and see practical first steps.',
    'conf': 'AI confidence score',
    'prob': 'This score shows model confidence, not a guaranteed diagnosis.',
    'uncertain': 'Check this result again',
    'uncertain_desc': 'The result is uncertain. Try a clearer photo or ask a local crop advisor.',
    'not_certain': 'Check the plant in the field before taking action.',
    'healthnote': 'This is initial guidance, not a confirmed diagnosis or pesticide instruction. Check field conditions and consult a local crop advisor before applying any products.',
    'lib_desc': 'Choose a condition to see common signs and practical next steps.',
    'history_title': 'Past screenings',
    'history_caption': 'Previous results are saved. Photos are not stored.',
    'how_desc': 'Use a bright, clear photo showing the affected part of the rice plant.',
    'top': 'Three possible results',
    'steps': 'Suggested next steps',
    'signs': 'Signs to check',
    'model_note': 'These figures come from model testing, not a guarantee for any single photo.',
})


def section_intro(eyebrow: str, title: str, detail: str = "", *, anchor: str = "") -> None:
    """Render an accessible, zero-height anchor inside the existing section header."""
    anchor_html = f'<span id="{escape(anchor)}" class="section-anchor"></span>' if anchor else ''
    st.markdown(
        anchor_html + '<div class="section-header">'
        f'<div class="section-eyebrow">{escape(eyebrow)}</div>'
        f'<h2>{escape(title)}</h2>'
        + (f'<p>{escape(detail)}</p>' if detail else '') + '</div>',
        unsafe_allow_html=True,
    )


def guide_content(label: str, lang: str, t: dict, *, expand_first: bool = True) -> None:
    """Keep agriculture instructions from backend.knowledge; only presentation changes."""
    advice = get_advice(label, lang)
    st.markdown(
        '<div class="knowledge-intro">'
        ''
        f'<h3>{escape(label)}</h3>'
        f'<p>{escape(advice["overview"])}</p></div>',
        unsafe_allow_html=True,
    )
    with st.expander(t['signs'], expanded=False):
        for item in advice['signs']:
            st.markdown(f'- {item}')
    st.markdown(f'<div class="panel-kicker" style="margin-top:22px">{escape(t["steps"])}</div>',
                unsafe_allow_html=True)
    stages = [('01', t['step1'], advice['initial']),
              ('02', t['step2'], advice['monitor']),
              ('03', t['step3'], advice['prevent'])]
    for number, title, items in stages:
        with st.expander(title, expanded=(expand_first and number == '01')):
            for item in items:
                st.markdown(f'- {item}')
    st.markdown('<div class="callout"><span class="note-icon" aria-hidden="true">i</span>'
                f'<span>{escape(t["healthnote"])}</span></div>', unsafe_allow_html=True)
    with st.expander(t['sources']):
        for ref in advice['sources']:
            st.markdown(f'- [{ref["title"]}]({ref["url"]})')


def display_percent(value: float, lang: str, digits: int = 1) -> str:
    number = f'{value * 100:.{digits}f}'
    return (number.replace('.', ',') if lang == 'id' else number) + '%'


def probability_rows(result: dict, lang: str) -> str:
    bars = []
    for label, confidence in sorted(result['probabilities'].items(),
                                    key=lambda pair: -pair[1])[:3]:
        percentage = max(0., min(100., float(confidence) * 100))
        bars.append(
            '<div class="prob-row"><div class="prob-heading">'
            f'<span>{escape(label)}</span><span>{display_percent(float(confidence), lang)}</span></div>'
            '<div class="prob-track">'
            f'<div class="prob-fill" style="width:{percentage:.1f}%" role="progressbar" '
            f'aria-label="{escape(label)}" aria-valuemin="0" aria-valuemax="100" '
            f'aria-valuenow="{percentage:.1f}"></div></div></div>'
        )
    return ''.join(bars)


# Cloud demo loads EfficientNet-B0 locally; keep API mode for the final-project backend.
MODEL_FILE = Path(__file__).resolve().parents[1] / "models" / "paddycare_model.pth"
DIRECT_INFERENCE = os.getenv("PADICARE_INFERENCE_MODE", "direct").lower() != "api"
BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("PADDYCARE_API_KEY", "")
PUBLIC_MODE = os.getenv("PADDYCARE_PUBLIC_MODE", "").lower() in {"1", "true", "yes"}
AUTH_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


# Native stateful toggle: OFF = Indonesia, ON = English.
# The track, moving thumb and ID/EN labels are styled above in CSS.
nav_left, nav_right = st.columns([5.4, 1.05], vertical_alignment='center', gap='small')
with nav_right:
    use_english = st.toggle(
        'English / Bahasa Inggris',
        value=False,
        key='language_toggle',
        label_visibility='collapsed',
    )
lang = 'en' if use_english else 'id'
t = TR[lang]

with nav_left:
    navigation = [
        ('beranda', 'Beranda', 'Home'), ('tentang', 'Tentang', 'About'),
        ('deteksi', 'Deteksi', 'Scan'), ('panduan', 'Panduan', 'Guide'),
        ('riwayat', 'Riwayat', 'History'),
    ]
    links = ''.join(
        f'<a href="#{anchor}" class="{ "menu-cta" if anchor == "deteksi" else "" }">'
        f'{escape(label_id if lang == "id" else label_en)}</a>'
        for anchor, label_id, label_en in navigation
    )
    st.markdown(
        '<nav class="site-nav" aria-label="Page navigation">'
        '<a class="site-brand" href="#beranda"><span class="site-symbol">'
        '<svg viewBox="0 0 32 32" aria-hidden="true">'
        '<path d="M8 25C8 11 18 7 26 7c-1 11-7 18-18 18Z" fill="none" '
        'stroke="#305f3d" stroke-width="2.5"/>'
        '<path d="M8 26c4-9 9-13 17-18" fill="none" stroke="#305f3d" stroke-width="2.4"/>'
        '</svg></span><span><span class="site-name">PadiCare AI</span>'
        '<span class="site-tag">Rice plant health</span></span></a>'
        f'<div class="site-links">{links}</div></nav>',
        unsafe_allow_html=True,
    )

headline = ('Kenali penyakit padi.<br><span>Rawat lebih awal.</span>' if lang == 'id' else
            'Spot rice plant problems.<br><span>Act early.</span>')
intro = ('Cukup unggah foto tanaman padi. Lihat hasil pemeriksaan dan langkah yang bisa dilakukan.'
         if lang == 'id' else
         'Upload a rice plant photo to check for possible issues and learn what to do next.')
cta = 'Mulai periksa' if lang == 'id' else 'Check a plant'

st.markdown(
    '<span id="beranda"></span><section class="hero-photo">'
    '<div class="hero-content">'
    f'<div class="hero-kicker">{ "PadiCare AI · Untuk petani padi" if lang == "id" else "PadiCare AI · For rice growers" }</div>'
    f'<h1>{headline}</h1><p>{escape(intro)}</p>'
    f'<a href="#deteksi" class="hero-cta">{escape(cta)} <span aria-hidden="true">↗</span></a>'
    '</div>'
    f'<div class="hero-caption">{ "Pemeriksaan awal dari foto, bukan pengganti pengamatan di sawah" if lang == "id" else "A photo-based first check, not a substitute for field inspection" }</div>'
    '</section>', unsafe_allow_html=True,
)

# Research results appear once, as a slim factual strip rather than oversized cards.
metrics = [
    ('10', 'jenis kondisi' if lang == 'id' else 'conditions'),
    ('93,66%' if lang == 'id' else '93.66%',
     'ketepatan pada data uji' if lang == 'id' else 'test accuracy'),
    ('93,57%' if lang == 'id' else '93.57%',
     'skor F1 data uji' if lang == 'id' else 'test F1 score'),
]
st.markdown('<div class="proof-strip">' + ''.join(
    f'<div class="proof-item"><b>{escape(number)}</b><span>{escape(label)}</span></div>'
    for number, label in metrics
) + '</div>', unsafe_allow_html=True)

# Three short steps are enough; no nested promotional cards.
steps = ([('01', 'Foto tanaman', 'Ambil foto yang jelas.'),
          ('02', 'Lihat hasil', 'Kenali kondisi yang mungkin terjadi.'),
          ('03', 'Lakukan langkah awal', 'Cek gejala dan ikuti panduan.')]
         if lang == 'id' else
         [('01', 'Take a photo', 'Use a clear photo.'),
          ('02', 'See the result', 'Review possible conditions.'),
          ('03', 'Take the first step', 'Check signs and read the guide.')])
steps_html = ''.join(
    f'<div class="process-item"><span class="process-no">{number}</span>'
    f'<div class="process-title">{escape(title)}</div><p>{escape(desc)}</p></div>'
    for number, title, desc in steps
)
about_title = ('Dari foto, jadi langkah awal.' if lang == 'id' else 'From a photo to a first step.')
about_description = (
    'PaddyCare membantu mengenali kemungkinan masalah pada tanaman padi. '
    'Periksa lagi keadaan tanaman sebelum menentukan tindakan.'
    if lang == 'id' else
    'PaddyCare helps identify possible rice plant issues. '
    'Check field conditions before deciding what to do.'
)
st.markdown(
    '<span id="tentang"></span><section class="editorial">'
    '<div class="section-header">'
    f'<div class="section-eyebrow">{ "Cara kerja" if lang == "id" else "How it works" }</div>'
    f'<h2>{escape(about_title)}</h2><p>{escape(about_description)}</p></div>'
    f'<div class="process">{steps_html}</div></section>',
    unsafe_allow_html=True,
)

# Detection: native Streamlit widgets, unchanged API contract.
with st.container(key='deteksi'):
    section_intro(
        'Pemeriksaan foto' if lang == 'id' else 'Photo check',
        'Periksa kondisi padi Anda' if lang == 'id' else 'Check your rice plant',
        t['how_desc'], anchor='deteksi',
    )
    left, right = st.columns([1,1.05], gap='large')
    with left:
        st.markdown('<div class="panel-kicker">01 / '+('Foto tanaman' if lang == 'id' else 'Plant photo')+'</div>', unsafe_allow_html=True)
        photo_method = st.radio(t['input_method'], [t['upload'],t['camera']],
                                horizontal=True, label_visibility='collapsed', key='image_source')
        if photo_method == t['upload']:
            upload = st.file_uploader(t['upload_label'], type=['jpg','jpeg','png','webp'], key='upload')
        else:
            upload = st.camera_input(t['photo_label'], key='camera')
        st.caption(t['help_photo'])
        if upload is not None:
            st.image(upload, use_container_width=True)
        submitted = st.button(t['analyze'], type='primary', use_container_width=True,
                              disabled=upload is None)
        if submitted and upload is not None:
            content = upload.getvalue()
            if len(content) > 8 * 1024 * 1024:
                st.error(t['help_photo'])
            else:
                try:
                    with st.spinner(t['wait']):
                        if DIRECT_INFERENCE:
                            result_data = predict_on_streamlit(content)
                        else:
                            resp = requests.post(
                                f"{BASE_URL}/predict", headers=AUTH_HEADERS,
                                files={"file": (upload.name or "photo.jpg", content,
                                                upload.type or "image/jpeg")},
                                timeout=120,
                            )
                            resp.raise_for_status()
                            result_data = resp.json()
                        st.session_state["last_result"] = result_data
                        st.session_state["last_upload_hash"] = hashlib.sha256(content).hexdigest()
                except Exception:
                    log.exception("Photo analysis failed")
                    st.error("Foto gagal dianalisis. Pastikan file model tersedia dan gambar valid."
                             if lang == "id" else
                             "Could not analyze this photo. Check the model file and image format.")
        # Never show a result from a different photo.
        if upload is None or (upload is not None and
                              hashlib.sha256(upload.getvalue()).hexdigest() !=
                              st.session_state.get('last_upload_hash')):
            st.session_state.pop('last_result', None)
        try:
            health = api_get('/health', timeout=3)
            if not health.get('model_file_exists'):
                st.warning(t['api_no_model'])
        except (requests.RequestException, ValueError):
            st.caption(t['api_down'] + ' · ' + t['api_help'])

    with right:
        st.markdown('<div class="panel-kicker">02 / '+('Hasil pemeriksaan' if lang == 'id' else 'Screening result')+'</div>', unsafe_allow_html=True)
        result = st.session_state.get('last_result')
        if result is None:
            st.markdown(
                '<div class="result-empty"><h3>'+
                ('Hasil akan tampil di sini' if lang == 'id' else 'Your result will appear here')+
                '</h3><p>'+
                ('Unggah foto yang jelas, lalu pilih Analisis Foto. Nama kondisi dan saran awal akan muncul setelah pemeriksaan.'
                 if lang == 'id' else
                 'Upload a clear photo and select Analyze Photo to see a possible condition and practical guidance.')+
                '</p></div>', unsafe_allow_html=True,
            )
        else:
            label = result['label']  # Always use the original model class name.
            confidence = float(result['confidence'])
            local_condition_name = get_advice(label, lang)['name']
            ranking = sorted(result['probabilities'].items(), key=lambda x: -x[1])
            difference = confidence - float(ranking[1][1]) if len(ranking) > 1 else confidence
            uncertain = confidence < .65 or difference < .15  # Display hint, not a calibrated threshold.
            st.markdown(
                '<div class="diagnosis-panel"><div class="diagnosis-kicker">'+escape(t['result'])+'</div>'
                f'<div class="diagnosis-code">{escape(label)}</div>'
                f'<div class="diagnosis-explain">{escape(local_condition_name)}</div>'
                f'<div class="diagnosis-score">{display_percent(confidence, lang)}</div>'
                '<div class="diagnosis-foot">'+escape(t['conf'])+'</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(t['prob'])
            if uncertain:
                st.warning(t['uncertain_desc'])
            else:
                st.markdown('<div class="callout"><span class="note-icon" aria-hidden="true">i</span>'
                            f'<span>{escape(t["not_certain"])}</span></div>', unsafe_allow_html=True)
            if label == 'normal':
                st.warning(t['warning_normal'])
            st.markdown(f'**{t["top"]}**')
            st.markdown(probability_rows(result, lang), unsafe_allow_html=True)
            st.download_button(
                t['download'], data=report_text(result,lang,t).encode('utf-8'),
                file_name=f'paddycare_{result["id"][:8]}.txt',
                mime='text/plain', use_container_width=True,
            )

# Show recommendations only after the user scans a photo.
if st.session_state.get('last_result') is not None:
    with st.container(key='saran'):
        section_intro('Setelah pemeriksaan' if lang=='id' else 'After the check',
                      'Apa yang bisa dilakukan sekarang?' if lang=='id' else 'What can you do now?')
        guide_content(st.session_state['last_result']['label'], lang, t)

# A compact knowledge guide instead of a long second set of recommendations.
with st.container(key='panduan'):
    section_intro('Tentang penyakit padi' if lang=='id' else 'Rice plant conditions',
                  'Panduan kondisi tanaman' if lang=='id' else 'Plant condition guide',
                  t['lib_desc'], anchor='panduan')
    chosen = st.selectbox(t['pick'], CLASSES, key='guide_class', format_func=lambda value:value)
    guide_content(chosen, lang, t, expand_first=False)

# Demo history is isolated per browser session and is not permanent in Cloud.
if DIRECT_INFERENCE:
    TR['id']['history_caption'] = 'Hanya pemeriksaan dari sesi ini. Riwayat hilang saat sesi berakhir.'
    TR['en']['history_caption'] = 'Only checks from this session. History is cleared when the session ends.'
    t = TR[lang]

with st.container(key='riwayat'):
    section_intro('Pemeriksaan sebelumnya' if lang=='id' else 'Earlier checks',
                  t['history_title'], t['history_caption'], anchor='riwayat')
    st.button(t['refresh'], key='refresh_history')
    try:
        history = [] if PUBLIC_MODE else api_get('/history?limit=100')
        stats = api_get('/stats')
        if PUBLIC_MODE:
            st.caption(t['public_history'])
        distribution = stats.get('distribution', {})
        c1,c2,c3 = st.columns(3)
        c1.metric(t['total'], int(stats.get('total',0)))
        c2.metric(t['found'], sum(int(n)>0 for n in distribution.values()))
        c3.metric(t['latest'], history[0]['label'] if history else '—')
        if history:
            with st.expander(t['hist_list'], expanded=False):
                rows = [{t['created']:r['created_at'][:19].replace('T',' '),
                         t['class']:r['label'],t['score']:f"{float(r['confidence'])*100:.1f}%",
                         t['id']:r['id'][:8]} for r in history]
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
                buf=io.StringIO()
                writer=csv.DictWriter(buf,fieldnames=['id','created_at','label','confidence'])
                writer.writeheader()
                writer.writerows(history)
                st.download_button(t['csv'],buf.getvalue().encode('utf-8-sig'),
                                   file_name='paddycare_history.csv',mime='text/csv')
        if int(stats.get('total',0))>0:
            with st.expander(t['hist_chart'],expanded=False):
                frame=pd.DataFrame([{'Condition':k,'Count':v} for k,v in distribution.items() if v>0])
                if not frame.empty:
                    st.bar_chart(frame.set_index('Condition'),horizontal=True,color='#50865e')
        elif not PUBLIC_MODE:
            st.caption(t['empty'])
    except (requests.RequestException, ValueError) as exc:
        st.caption(f"{t['errors']}: {exc}")

# Short footer with a photo credit matching the asset in use.
photo_credit = (
    'Foto: aset lokal' if lang == 'id' else 'Photo: local asset'
) if HERO_URL.startswith('data:') else (
    'Photo: <a href="https://unsplash.com/photos/hqnUYXsN5oY" target="_blank" '
    'rel="noopener noreferrer">Ivan Bandura / Unsplash</a>'
)
st.markdown(
    '<div class="footer"><span>© PadiCare AI · EfficientNet-B0 · FastAPI · Streamlit</span>'
    '<span><a href="https://rice-diseases.irri.org/" target="_blank" '
    'rel="noopener noreferrer">IRRI Rice Diseases</a> · '
    + photo_credit + '</span></div>',
    unsafe_allow_html=True,
)

# Smoothly navigate through the existing single-page sections.
# The component does not replace the navbar or affect model/API logic.
components.html(r"""
<script>
try {
  const win = window.parent;
  const doc = win.document;
  const anchors = new Set(['beranda', 'tentang', 'deteksi', 'panduan', 'riwayat']);

  // Only one navigation handler should exist after Streamlit reruns.
  if (win.__paddyCareSmoothScrollV56) {
    win.__paddyCareSmoothScrollV56.controller.abort();
    win.__paddyCareSmoothScrollV56.navObserver.disconnect();
    delete win.__paddyCareSmoothScrollV56;
  }
  if (!win.__paddyCareSmoothScrollV57) {
    const controller = new win.AbortController();
    const signal = controller.signal;
    const nav = () => doc.querySelector('[data-testid="stHorizontalBlock"]:has(.site-nav)');
    const allScrollers = () => [
      doc.querySelector('[data-testid="stAppViewContainer"]'),
      doc.querySelector('[data-testid="stMain"]'),
      doc.querySelector('section.main'),
      doc.scrollingElement,
    ].filter(Boolean);

    const scrollAmount = () => Math.max(win.scrollY || 0, ...allScrollers().map(el => el.scrollTop || 0));
    const updateNavbar = () => nav()?.classList.toggle('nav-scrolled', scrollAmount() > 65);

    function scrollContainerFor(target) {
      // The scrolling element differs across Streamlit versions.
      for (let node = target.parentElement; node && node !== doc.body; node = node.parentElement) {
        const style = win.getComputedStyle(node);
        if (/(auto|scroll|overlay)/.test(style.overflowY) &&
            node.scrollHeight > node.clientHeight + 5) return node;
      }
      return doc.scrollingElement || doc.documentElement;
    }

    let animation = null;
    let activeScroller = null;
    let originalScrollBehavior = '';

    function stopAnimation() {
      if (animation !== null) win.cancelAnimationFrame(animation);
      animation = null;
      if (activeScroller) {
        activeScroller.style.scrollBehavior = originalScrollBehavior;
        activeScroller = null;
      }
    }

    function navigateTo(id) {
      const target = doc.getElementById(id);
      if (!target) return;
      stopAnimation();

      const scroller = scrollContainerFor(target);
      const isPage = scroller === doc.scrollingElement || scroller === doc.documentElement;
      const scrollerTop = isPage ? 0 : scroller.getBoundingClientRect().top;
      const navBottom = nav()?.getBoundingClientRect().bottom || 0;
      const offset = Math.max(112, navBottom - scrollerTop + 16);
      const current = scroller.scrollTop;
      const destination = id === 'beranda' ? 0 :
        Math.max(0, Math.min(scroller.scrollHeight - scroller.clientHeight,
          current + target.getBoundingClientRect().top - scrollerTop - offset));

      if (win.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        scroller.scrollTop = destination;
        updateNavbar();
        return;
      }

      // Start moving immediately, then slow down near the destination.
      // Temporarily disable CSS smooth scrolling so it does not fight
      // the requestAnimationFrame movement on every frame.
      const distance = destination - current;
      if (Math.abs(distance) < 2) return;
      activeScroller = scroller;
      originalScrollBehavior = scroller.style.scrollBehavior;
      scroller.style.scrollBehavior = 'auto';

      // Short sections still feel deliberate; long jumps do not drag on.
      const duration = Math.min(480, Math.max(280, 240 + Math.abs(distance) * 0.065));
      let started = null;
      function tick(time) {
        if (started === null) started = time;
        const t = Math.min(1, (time - started) / duration);
        const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic, no slow start
        scroller.scrollTop = current + distance * eased;
        updateNavbar();
        if (t < 1) {
          animation = win.requestAnimationFrame(tick);
        } else {
          scroller.scrollTop = destination;
          stopAnimation();
        }
      }
      animation = win.requestAnimationFrame(tick);
    }

    win.addEventListener('click', event => {
      const link = event.target.closest('a[href^="#"]');
      if (!link) return;
      const id = link.getAttribute('href').slice(1);
      if (!anchors.has(id) || !doc.getElementById(id)) return;
      event.preventDefault();
      // Prevent earlier Streamlit rerun handlers from jumping immediately.
      event.stopPropagation();
      navigateTo(id);
      win.history.replaceState(null, '', '#' + id);
    }, {capture: true, signal});

    // Let the visitor interrupt the animation to scroll manually.
    ['wheel', 'touchstart', 'pointerdown'].forEach(name =>
      doc.addEventListener(name, stopAnimation, {passive: true, signal}));
    doc.addEventListener('keydown', event => {
      if (['PageUp', 'PageDown', 'ArrowUp', 'ArrowDown', 'Home', 'End', ' '].includes(event.key))
        stopAnimation();
    }, {signal});

    let listeningOn = [];
    function bindScrollListeners() {
      const next = allScrollers();
      listeningOn.filter(el => !next.includes(el)).forEach(el => el.removeEventListener('scroll', updateNavbar));
      next.filter(el => !listeningOn.includes(el)).forEach(el =>
        el.addEventListener('scroll', updateNavbar, {passive: true}));
      listeningOn = next;
      updateNavbar();
    }
    win.addEventListener('scroll', updateNavbar, {passive: true, signal});
    const navObserver = new win.MutationObserver(bindScrollListeners);
    navObserver.observe(doc.body, {childList: true, subtree: true});
    bindScrollListeners();
    win.__paddyCareSmoothScrollV57 = {controller, navObserver};
  }

  // Reveal headings as the visitor reaches them; content remains visible
  // if reduced motion is enabled or IntersectionObserver is unavailable.
  if (!win.__paddyCareSectionReveal &&
      'IntersectionObserver' in win &&
      !win.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const selector = '.section-header, .editorial .process-item';
    const io = new win.IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('in-view');
        io.unobserve(entry.target);
      });
    }, {threshold: .08, rootMargin: '0px 0px -45px 0px'});
    const register = () => {
      doc.querySelectorAll(selector).forEach(el => {
        if (el.classList.contains('reveal-ready')) return;
        el.classList.add('reveal-ready');
        io.observe(el);
      });
    };
    register();
    doc.documentElement.classList.add('paddy-motion');
    const mo = new win.MutationObserver(register);
    mo.observe(doc.body, {childList: true, subtree: true});
    win.__paddyCareSectionReveal = {io, mo};
  }
} catch (_) {
  // Native HTML anchors and CSS smooth scrolling remain as a fallback.
}
</script>
""", height=0)
