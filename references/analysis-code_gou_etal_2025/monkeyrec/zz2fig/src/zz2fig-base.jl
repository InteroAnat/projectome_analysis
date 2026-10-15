function rasterize_pdf(pdf, outpdf; hiq=false)
    dpi = hiq ? 250 : 200
    qual = hiq ? 97 : 92
    mktempdir() do path
        bmp = joinpath(path, "tmp.png")
        #run(`pdftocairo -r 240 -jpeg -singlefile -jpegopt 'quality=100,optimize=y' $pdf $pdf.jpg`)
        #run(`pdftocairo -r $dpi -png -singlefile $pdf $pdf.png`)
        run(`pdftocairo -r $dpi -png -singlefile $pdf $(splitext(bmp)[1])`)
        #run(`convert -verbose -density $dpi -trim $bmp -compress jpeg2000 -quality $qual -flatten $pdf.xxx.pdf`)
        #run(`convert -verbose -density 240 -trim $pdf -quality 100 -flatten -sharpen 0x1.0 $bmp`)
        #cp(bmp, "$outpdf.jpg", force=true)
        #run(`convert -verbose -density $dpi -trim $bmp -compress jpeg -quality $qual -flatten -sharpen 0x1.0 $outpdf`)
        run(`convert -verbose -density 78.74 -trim $bmp -compress jpeg -quality $qual -flatten $outpdf`)
        #run(`convert -verbose -density $dpi -trim $bmp -compress jpeg2000 -quality $qual -flatten $outpdf`)
    end
    filesize(pdf) > filesize(outpdf) && return outpdf
    pdf
end

function join_pdf(pdfs, outpdf; hiq=false)
    mktempdir() do path
        temps = map(pdfs) do f
            of = joinpath(path, basename(f))
            rasterize_pdf(f, of; hiq)
        end
        run(`pdfunite $temps $outpdf`)
    end
end

