﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class UI_Count : MonoBehaviour
{

    public Text txt;
    int cnt = 0;
    public int amountCount = 1;
    public Slider myslider;
    private void Start()
    {
        cnt = 0;
    }
    public void CountAdd()
    {
        cnt += amountCount;
        txt.text = ("" + cnt);
    }
    public void CountSub()
    {
        cnt -= amountCount;
        txt.text = ("" + cnt);
    }
}