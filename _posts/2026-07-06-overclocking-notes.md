---
layout:       post
title:        "Overclocking notes: 9700X + DDR5 2x24GB 8400MHz"
date:         2026-07-06
tags:         tips
---


I recently upgraded my desktop from [Intel i7-7800X](https://www.intel.com/content/www/us/en/products/sku/123589/intel-core-i77800x-xseries-processor-8-25m-cache-up-to-4-00-ghz/specifications.html) to [AMD 9700X](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-7-9700x.html), because my old CPU was really hot and made the entire room very [hot](({% link _posts/2026-06-28-hot-summer-in-switzerland.md %})) in the [summer]({% link _posts/2026-06-28-hot-summer-in-switzerland.md %}). It's my first time to use an AMD CPU and DDR5 RAM, so I wrote this note for my future reference. This can be **wrong** as my main source is reddit posts, some of which are contradicting.


## My hardware

* motherboard: [TUF GAMING B850-PLUS WIFI](https://www.asus.com/ch-en/motherboards-components/motherboards/tuf-gaming/tuf-gaming-b850-plus-wifi/)
* CPU: [AMD Ryzen 7 9700X](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-7-9700x.html)
* RAM: [Kingston FURY Renegade DDR5 8400MT/s 2$\times$24GB](https://www.kingston.com/en/memory/gaming/fury-renegade-ddr5-rgb?module%20type=cudimm&speed=8400mt%2Fs&total%20(kit)%20capacity=24gb&kit=single%20module&dram%20density=24gbit&color=black) (second hand)


## Why overclocking?

Because my memory sticks can't work efficiently at the default setting. To make them work faster, we need to adjust the voltage(s), which at the same time would affect CPU, so it means we need to tuning CPU and RAM together.

In DDR4 era, all we need is often to plug the RAM in, and it works at the advertised frequency. Done. However, DDR5 doesn't work like that. It usually have two or more timing parameters: a [default JEDEC one](https://www.lttlabs.com/articles/2026/02/26/who-is-high-speed-ddr5-memory-actually-for#tests-and-results), and some XMP and/or EXPO profiles. For example, my RAM is advertised to run at *up to* 8400MT/s, with [default JEDEC frequency](https://www.kingston.com/datasheets/KF584CU40RSA-24.pdf) only at 6400MT/s. Without tuning the BIOS, the RAM will run at the default 6400MT/s CL52-52-52, with is significantly slow because it trades performance for stability (in a non-Pareto sense in my opinion). In my case, the actual default is even *worse* than its JEDEC rate: it's 5600MT/s CL 46-45-45. This is huge performance loss.

After tuning the RAM, CPU will be affected due to a higher voltage. I'm not interested in overclocking CPU per se, but under-voltaging it does benefit temperature, which is a main reason why I upgraded my rig.


## Things to tune

**Starting Point.** Running RAM at a higher frequency needs higher voltages. DDR5 RAM often carries some "performance" profiles. These configurations set voltages, frequency, timing and everything for us, and are often a good starting point. Depending on the hardware, these settings may be called [**XMP**](https://www.intel.com/content/www/us/en/gaming/extreme-memory-profile-xmp.html), [**EXPO**](https://www.amd.com/en/products/processors/technologies/expo.html), or [**DOCP**](https://www.asus.com/ch-en/support/faq/1042256/) in BIOS. Before changing the profile, keep in mind that it may [void AMD products warranty](https://www.amd.com/en/products/processors/technologies/expo.html). [ASUS AM5 motherboard warranty](https://www.asus.com/us/news/ihctikmgahafyrib/) *does* cover these memory configurations, but it's motherboard, not including CPU.

My RAM has three profiles: 8400 CL40-52-52, 8000 CL38-48-48, and 7600 CL38-46-46. The 8400MT/s one can't boot (even if it falls inside [support range](https://www.asus.com/motherboards-components/motherboards/tuf-gaming/tuf-gaming-b860-plus-wifi/techspec/)). The 8000MT/s can boot and passes stability tests, but occasionally (like once a week) the desktop just freezes without BSOD at normal/low workload. The 7600MT/s profile is the only usable one.

**Lower the Voltages.** First, both [**VDDCR_SOC**](https://www.asus.com/ch-en/motherboards-components/motherboards/tuf-gaming/tuf-gaming-b850-plus-wifi/helpdesk_manual?model2Name=TUF-GAMING-B850-PLUS-WIFI) and [**VDDIO / MC**](https://www.asus.com/ch-en/motherboards-components/motherboards/tuf-gaming/tuf-gaming-b850-plus-wifi/helpdesk_manual?model2Name=TUF-GAMING-B850-PLUS-WIFI) feed memory controller and scales with memory overclocking. In fact, they [control different voltages](https://www.techpowerup.com/forums/threads/ram-expo-profile-at-1-45v-normal.339690/post-5567344) but anyways both of them should be tuned. Second, [**VDD**](https://www.asus.com/ch-en/motherboards-components/motherboards/tuf-gaming/tuf-gaming-b850-plus-wifi/helpdesk_manual?model2Name=TUF-GAMING-B850-PLUS-WIFI) and [**VDDQ**](https://www.asus.com/ch-en/motherboards-components/motherboards/tuf-gaming/tuf-gaming-b850-plus-wifi/helpdesk_manual?model2Name=TUF-GAMING-B850-PLUS-WIFI) power memory chips. Finally, as VDDCR_SOC powers the memory controller, which is physically inside the CPU package, changing it will have some (small) impact on CPU temperature. So we can lower CPU voltage through [**Curve Optimizer**](https://www.amd.com/content/dam/amd/en/documents/products/software-tools/faq-curve-optimizer.pdf) and [**CPU SoC** and **core voltage**](https://www.amd.com/content/dam/amd/en/documents/products/software-tools/ryzen-master-quick-reference-guide.pdf), and lower the temperature.[^change-the-curve] I'm not interested in overclocking the CPU itself, so I didn't explore anything beyond this.

[^change-the-curve]: Curve Optimizer does not hard code the voltage reduction. Instead, it changes the factory-calibrated voltage-frequency curve. This curve tells CPU how many voltages it should request at any frequency. Curve Optimiser shifts the entire curve down.


## Things to monitor

Generally speaking, for all voltages above, we want them to be low, as long as the computer is stable, and temperature is under control. So three things to monitor: voltage not too high, temperature not too hot, and system stability. [HWiNFO](https://www.hwinfo.com/) and [OCCT](https://www.ocbase.com/) are good tools for this.

### Voltage

|                                                                                        | For what                                             | Dangerous limit (V)                                                                                                                                                                 |
|----------------------------------------------------------------------------------------|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CPU**                                                                                |                                                      |                                                                                                                                                                                      |
| VDDCR_SOC (SVI3 TFN)                                                                   | Powers CPU SoC (memory controller, PCIe lanes, etc.) | 1.3 ([AMD](https://www.amd.com/en/resources/support-articles/release-notes/RN-RYZEN-MASTER-2-10-3-2518.html) and [ASUS](https://www.asus.com/au/support/faq/1050307/) safety limit) |
| [VDDCR_VDD](https://docs.amd.com/r/en-US/68886-ryzen-master-user-guide/CPU) (SVI3 TFN) | CPU core voltage                                     | 1.4? (couldn't find good source. Some [posts](https://www.reddit.com/r/Amd/comments/13bj8w1/cpu_vddcr_vdd_and_vddcr_soc/) says 1.4V is safe                                         |
| VDDIO / MC                                                                             | Powers memory controller                             | 1.5? (again couldn't find good source. Use value from [this](https://www.reddit.com/r/overclocking/comments/1q5k6ua/vddiomc/))                                                       |
| **Memory**                                                                             |                                                      |                                                                                                                                                                                      |
| [VDD](https://www.techpowerup.com/articles/45)                                         | Powers the memory                                    | [1.45](https://forum-en.msi.com/index.php?threads/safe-voltage-range-for-cpu-vddq-and-cpu-vdd2.383716/)                                                                             |
| [VDDQ](https://www.techpowerup.com/articles/45)                                        | Powers the communication between CPU and memory      | [1.45](https://forum-en.msi.com/index.php?threads/safe-voltage-range-for-cpu-vddq-and-cpu-vdd2.383716/)                                                                             |

### Temperature

|                     | What                                  | Dangerous limit (℃)                                                                                                                                                                         |
|---------------------|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CPU**             |                                       |                                                                                                                                                                                             |
| Tctl/Tdie           | Hotspot temperature of the entire CPU | 95 (AMD [hard limit](https://www.techpowerup.com/review/amd-ryzen-7-9700x/25.html))                                                                                                         |
| **Memory**          |                                       |                                                                                                                                                                                             |
| SPD hub temperature | Memory temperature                    | 85 as [official limit](https://www.kingston.com/datasheets/KF584CU40RSA-24.pdf), but a more [realistic limit](https://www.techpowerup.com/review/ddr5-temperature-variable-analysis/) is 45 |


### Performance and stability

I use [AIDA64](https://www.aida64.com/user-manual/benchmarks/cache-memory-benchmark) to measure memory bandwidth and latency, and use [Cinebench](https://www.maxon.net/en/cinebench) for CPU performance. For stability, I use [OCCT](https://www.ocbase.com/) and [y-cruncher](https://www.numberworld.org/y-cruncher/) with this [config](https://github.com/integralfx/MemTestHelper/blob/oc-guide/DDR4%20OC%20Guide.md).[^memtest-config]

[^memtest-config]: You probably need to adjust the test items to run, depending on the y-cruncher version.


## How to tune

I started with the built-in DOCP 7600 CL38-46-46 DOCP profile. It passes stability test. Then I tried to lower all voltages [above](#things-to-tune) one by one, until it fails to boot or the stability test. I finally arrived at the following BIOS settings:

* integrated GPU: disabled (as I have a proper GPU)
* Curve Optimizer: -25 offset
* CPU Core Voltage: -0.15 offset
* CPU SOC Voltage: -0.15 offset
* CPU VDDIO / MC Voltage: 1.35
* DRAM VDD Voltage: 1.35
* DRAM VDDQ Voltage: 1.35

All voltages and temperatures are below the dangerous level, and it has been stable for a couple of days. Relative to default, Cinebench benchmark gets better by 20+%, and memory latency drops 30+%.

### Some weird things

Claude and many posts insist that running the memory at a lower frequency brings lower latency. I *can't* reproduce this on my machine. I tried 5600 and 6000MT/s with many timing and voltage settings. None of them is better than the 7600MT/s DOCP profile. I'm not sure why.
