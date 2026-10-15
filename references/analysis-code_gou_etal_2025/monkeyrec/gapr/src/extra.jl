Make.@target stat_filesize_reduction (snapshots2) ->begin
	fn=cache_file_swcdir("18925", 49)
	offsets=Set()
	function fmt(v)
		r="$v"
		repeat("0", 8-length(r))*r
	end
	pat1(x, y, z)="ch00webm/z$(fmt(z))/y$(fmt(y)).x$(fmt(x)).webm"
	pat2(x, y, z)="ch00/z$(fmt(z))/y$(fmt(y)).x$(fmt(x)).nrrd"
	for f in readdir(fn, join=true)
		match(r".*\.swc(\.gz)?$", f)!==nothing || continue
		open(f, "r") do io
			while !eof(io)
				l=readline(io)
				v=split(l, " ")
				length(v)==7 || continue
				x=parse(Float64, v[3])
				y=parse(Float64, v[4])
				z=parse(Float64, v[5])
				x=floor(Int64, x/.35/384)*384
				y=floor(Int64, y/.35/384)*384
				z=floor(Int64, z/1/128)*128
				push!(offsets, (x, y, z))
			end
		end
	end
	fsize=Dict{String,Int64}()
	open("./precious/stat-imgfile-18925.txt", "r") do io
		while !eof(io)
			l=readline(io)
			m=match(r"^(.*) (\d+)$", l)
			m!==nothing || continue
			s=parse(Int64, m.captures[2])
			fsize[m.captures[1]]=s
		end
	end

	df=map([("webm", pat1), ("nrrd", pat2)]) do (typ, pat)
		ss=0
		for x in 0:384:29399
			for y in 0:384:18448
				for z in 0:128:11623
					ss+=get(fsize, pat(x, y, z), nothing)
				end
			end
		end
		tt=0
		for (x, y, z) in offsets
			tt+=get(fsize, pat(x, y, z), nothing)
		end
		(format=typ, total=round(ss/1000/1000/1000), used=round(tt/1000/1000/1000))
	end
	DataFrame(df)
end

