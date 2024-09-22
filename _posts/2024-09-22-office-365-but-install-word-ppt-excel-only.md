---
layout:       post
title:        "Office 365, but install Word, PowerPoint, and Excel only"
date:         2024-09-22
tags:         tips
---


I bought a used desktop recently and am setting up the system. One very annoying thing is that when you install Office 365, it includes the entire suite by default. That is, we get OneNote and other useless stuff. I do *not* want them! Word, Excel, and PowerPoint, and nothing else! It's ridiculous that we can't uninstall an individual app after installation.

> In most cases you can't uninstall an individual app if it's included in your Microsoft 365 suite. The only way to uninstall an individual app is if you purchased it as a stand-alone app.
>
> <div style="text-align: right">—<a href="https://support.microsoft.com/en-us/office/uninstall-office-from-a-pc-9dd49b83-264a-477a-8fcc-2fdf5dbf61d8#OfficeVersion=Click-to-Run_or_MSI">Uninstall Office from a PC</a></div>

So in order to have only World, Excel, and PowerPoint, we need to make sure all other Office 365 apps are excluded during installation. This is possible through [Office Deployment Tool](https://www.microsoft.com/en-us/download/details.aspx?id=49117) (ODT). After downloading ODT, you should see a `setup.exe` and a few `.xml` files, which are config templates. You can modify whichever `.xml` or create your own one from scratch. Just make sure your Office version, e.g. x86 or x64, is correctly specified. I created this `configuration-Office365-x64.xml`:

```xml
<Configuration>
    <Add OfficeClientEdition="64" Channel="BetaChannel">
        <Product ID="O365ProPlusRetail">
            <Language ID="zh-cn"/>
            <Language ID="en-gb"/>
            <ExcludeApp ID="Access"/>
            <ExcludeApp ID="Groove"/>
            <ExcludeApp ID="OneDrive"/>
            <ExcludeApp ID="OneNote"/>
            <ExcludeApp ID="Outlook"/>
            <ExcludeApp ID="Publisher"/>
            <ExcludeApp ID="Lync"/>
            <ExcludeApp ID="Bing"/>
        </Product>
        <Product ID="ProofingTools">
            <Language ID="en-gb"/>
            <Language ID="en-us"/>
        </Product>
    </Add>
</Configuration>
```

You can go through [Overview of the Office Deployment Tool](https://learn.microsoft.com/en-us/microsoft-365-apps/deploy/overview-office-deployment-tool#exclude-or-remove-microsoft-365-apps-applications-from-client-computers) to understand what the configuration file is. There is also a webpage that you can create this file via GUI, at [Office Customization Tool - Microsoft 365 Apps admin center](https://config.office.com/deploymentsettings), if you have access to it.

The final step is to download and install Office using this config file in CMD:

```bash
setup.exe /download configuration-Office365-x64.xml
setup.exe /configure configuration-Office365-x64.xml
```

Done!
