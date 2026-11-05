' Secondary pattern of every flight-array element through the shaped reflector, computed with CST's
' asymptotic solver, for comparison with dbf_cfm.py (one run per element, polarization, frequency).
'
' Before running
'   1. python cst/make_asolver_jobs.py --ffs-dir EEP_ffs --out asolver
'      and put the "asolver" folder (jobs.csv, sources\, manifest.json) next to shaped_reflector.cst.
'   2. In shaped_reflector.cst: units m / GHz, reflector vertex at the origin opening toward +z,
'      reflector PEC, solver type Asymptotic with its settings (accuracy, ray options) chosen once
'      in the GUI. The array plane is at z = array_plane_z_m in manifest.json (about 5.606 m).
'   3. Set MAX_JOBS = 2 below for a first test, run (Macros > Edit/Run, or Home > Macros), check
'      asolver\results, then set MAX_JOBS = 0 and run again. Finished jobs are skipped, so the
'      macro can be stopped and restarted at any time.
'
' Output: asolver\results\job<N>.ffs (CST farfield source format, phase reference at the CST
' origin = reflector vertex, theta measured from +z, kept for theta <= THETA_KEEP) and
' asolver\results\log.txt. Row N of jobs.csv says which element, polarization and frequency it is.
Option Explicit

Const JOB_DIR = ""             ' empty: <project folder>\asolver\
Const SOURCE_NAME = "eep"
Const STEP_THETA = "0.2"       ' export grid [deg]
Const STEP_PHI = "1"
Const THETA_KEEP = 20          ' keep theta <= THETA_KEEP deg in the stored files (0: keep the full sphere)
Const MAX_JOBS = 0             ' 0: run all remaining jobs; e.g. 2 for a first test

Sub Main
	Dim jobDir As String, outDir As String, txt As String, f As Variant
	Dim fin As Integer, nDone As Long, nSkip As Long, t0 As Double

	jobDir = JOB_DIR
	If jobDir = "" Then jobDir = GetProjectPath("Root") & "\asolver\"
	If Right(jobDir, 1) <> "\" Then jobDir = jobDir & "\"
	outDir = jobDir & "results\"
	On Error Resume Next
	MkDir outDir
	ChangeSolverType "HF Asymptotic"
	On Error GoTo 0

	fin = FreeFile
	Open jobDir & "jobs.csv" For Input As #fin
	Line Input #fin, txt                                   ' header
	Do While Not EOF(fin)
		Line Input #fin, txt
		If Trim(txt) <> "" Then
			' job,element,i,j,eep_class,pol,port,freq_ghz,x_m,y_m,z_m,source_file
			f = Split(Trim(txt), ",")
			If Dir(outDir & "job" & f(0) & ".ffs") <> "" Then
				nSkip = nSkip + 1
			Else
				t0 = Timer
				RunJob f, jobDir, outDir
				nDone = nDone + 1
				WriteLog outDir, "job " & f(0) & " element " & f(1) & " " & f(5) & " " & f(7) & " GHz: " & _
					Format(Timer - t0, "0.0") & " s"
				If MAX_JOBS > 0 And nDone >= MAX_JOBS Then Exit Do
			End If
		End If
	Loop
	Close #fin
	MsgBox nDone & " jobs run, " & nSkip & " already done." & vbCrLf & "Results in " & outDir
End Sub

Sub RunJob(f As Variant, jobDir As String, outDir As String)
	Dim freq As String, item As String, tmp As String
	freq = Trim(f(7))

	On Error Resume Next                                   ' the monitor may already exist
	With Monitor
		.Reset
		.Name "farfield (f=" & freq & ")"
		.Domain "Frequency"
		.FieldType "Farfield"
		.Frequency freq
		.Create
	End With
	FarfieldSource.Delete SOURCE_NAME
	On Error GoTo 0

	With FarfieldSource
		.Reset
		.Name SOURCE_NAME
		.Id "1"
		.UseCopyOnly "false"
		.Setpos Trim(f(8)), Trim(f(9)), Trim(f(10))
		.Settheta "0", "0", "-1"                         ' element broadside (+z in the EEP file) -> -z
		.Setphi "1", "0", "0"                            ' element +x (H polarization) stays +x
		.Import jobDir & "sources\" & Trim(f(11))
		.Store
	End With

	StartAsymptoticSolver

	item = FindFarfield(freq)
	If item = "" Then Err.Raise 1000, , "no farfield result at " & freq & " GHz after job " & f(0)
	SelectTreeItem item
	tmp = outDir & "job" & f(0) & ".tmp"
	With FarfieldPlot
		.Reset
		.Plottype "3d"
		.SetPlotMode "efield"
		.Origin "free"
		.Userorigin "0", "0", "0"                       ' phase reference: reflector vertex
		.Step STEP_THETA
		.Step2 STEP_PHI
		.Plot
		.ASCIIExportAsSource tmp
	End With
	If THETA_KEEP > 0 Then
		TrimFfs tmp, outDir & "job" & f(0) & ".ffs", THETA_KEEP
		Kill tmp
	Else
		Name tmp As outDir & "job" & f(0) & ".ffs"
	End If
End Sub

' Start the asymptotic solver (CST macro object AsymptoticSolver; the older name ASolver does not
' compile in current versions).
Sub StartAsymptoticSolver()
	AsymptoticSolver.Start
End Sub

' Tree path of the farfield result at freq [GHz], e.g. "Farfields\farfield (f=1.3) [eep]".
Function FindFarfield(freq As String) As String
	Dim item As String, label As String, i As Long
	item = Resulttree.GetFirstChildName("Farfields")
	Do While item <> ""
		label = Mid(item, InStrRev(item, "\") + 1)
		i = InStr(label, "f=")
		If i > 0 Then
			If Abs(Val(Mid(label, i + 2)) - Val(freq)) < 0.000001 Then
				FindFarfield = item
				Exit Function
			End If
		End If
		item = Resulttree.GetNextItemName(item)
	Loop
	FindFarfield = ""
End Function

' Copy a farfield source file keeping only theta <= thetaMax, with the sample count fixed.
Sub TrimFfs(src As String, dst As String, thetaMax As Double)
	Dim a As Integer, b As Integer, s As String, nPhi As Long, nKeep As Long
	Dim inData As Boolean, wantCount As Boolean, countDone As Boolean

	a = FreeFile                                           ' pass 1: count the kept samples
	Open src For Input As #a
	Do While Not EOF(a)
		Line Input #a, s
		If wantCount And IsDataLine(s) Then
			nPhi = Val(Token(s, 0))
			wantCount = False
		ElseIf InStr(LCase(s), "phi samples") > 0 Then
			wantCount = True
		ElseIf InStr(LCase(s), ">> phi, theta") > 0 Then
			inData = True
		ElseIf inData And IsDataLine(s) Then
			If Val(Token(s, 1)) <= thetaMax + 0.000001 Then nKeep = nKeep + 1
		End If
	Loop
	Close #a

	inData = False: wantCount = False
	a = FreeFile                                           ' pass 2: write
	Open src For Input As #a
	b = FreeFile
	Open dst For Output As #b
	Do While Not EOF(a)
		Line Input #a, s
		If wantCount And IsDataLine(s) Then
			Print #b, nPhi & " " & (nKeep \ nPhi)
			wantCount = False
		ElseIf inData And IsDataLine(s) Then
			If Val(Token(s, 1)) <= thetaMax + 0.000001 Then Print #b, s
		Else
			If InStr(LCase(s), "phi samples") > 0 Then wantCount = True
			If InStr(LCase(s), ">> phi, theta") > 0 Then inData = True
			Print #b, s
		End If
	Loop
	Close #b
	Close #a
End Sub

Function IsDataLine(s As String) As Boolean
	Dim t As String
	t = Trim(Replace(s, vbTab, " "))
	IsDataLine = (t <> "" And Left(t, 2) <> "//")
End Function

' k-th (0-based) whitespace-separated token of a line.
Function Token(s As String, k As Long) As String
	Dim p As Variant, i As Long, n As Long
	p = Split(Trim(Replace(s, vbTab, " ")), " ")
	For i = 0 To UBound(p)
		If p(i) <> "" Then
			If n = k Then
				Token = p(i)
				Exit Function
			End If
			n = n + 1
		End If
	Next i
	Token = ""
End Function

Sub WriteLog(outDir As String, msg As String)
	Dim h As Integer
	h = FreeFile
	Open outDir & "log.txt" For Append As #h
	Print #h, Format(Now, "yyyy-mm-dd hh:nn:ss") & "  " & msg
	Close #h
	ReportInformationToWindow msg
End Sub
