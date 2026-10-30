' Export every embedded-element farfield in the result tree as a CST farfield source (.ffs),
' optionally also as an ASCII E-field table. Skips the array-factor results ("PAA_FA_FREQ").
' Run from CST: Macros > Edit/Run (or Home > Macros > Run Macro), with the project open.
'
' Output names: farfield_f<freq>_port<n>.ffs   e.g. farfield_f1.3_port31.ffs
' Check the header of the first file: Position, #Frequencies, power lines, theta/phi sample count.
Option Explicit

Const OUT_DIR = ""              ' empty: <project folder>\EEP_ffs\ ; or e.g. "C:\EEP\"
Const SKIP_TAG = "PAA_FA_FREQ"
Const EXPORT_ASCII_TOO = False  ' also write an ASCII table (Theta, Phi, |E|, E_theta, phase, E_phi, phase)
Const STEP_DEG = "1"            ' theta / phi step for the export grid

Sub Main
	Dim outDir As String, item As String, label As String, base As String
	Dim nDone As Long, nSkip As Long

	outDir = OUT_DIR
	If outDir = "" Then outDir = GetProjectPath("Root") & "\EEP_ffs\"
	If Right(outDir, 1) <> "\" Then outDir = outDir & "\"
	On Error Resume Next
	MkDir outDir
	On Error GoTo 0

	item = Resulttree.GetFirstChildName("Farfields")
	Do While item <> ""
		label = Mid(item, InStrRev(item, "\") + 1)
		If InStr(label, SKIP_TAG) > 0 Then
			nSkip = nSkip + 1
		ElseIf SelectTreeItem(item) Then
			base = outDir & FileLabel(label)
			With FarfieldPlot
				.Reset
				.Plottype "3d"                    ' .ffs export needs a 3D farfield plot
				.SetPlotMode "efield"
				.Origin "free"                    ' one common phase reference for every element:
				.Userorigin "0", "0", "0"         ' the CST global origin (de-embed per element later)
				.Step STEP_DEG
				.Step2 STEP_DEG
				.Plot
				.ASCIIExportAsSource base & ".ffs"
				If EXPORT_ASCII_TOO Then
					.ASCIIExportVersion "2.0"
					.ASCIIExport base & ".txt"
				End If
			End With
			nDone = nDone + 1
			ReportInformationToWindow "exported " & label
		End If
		item = Resulttree.GetNextItemName(item)
	Loop

	MsgBox nDone & " farfields exported to " & outDir & vbCrLf & nSkip & " skipped (" & SKIP_TAG & ")"
End Sub

' "farfield (f=1.3) [31]"  ->  "farfield_f1.3_port31"
Function FileLabel(label As String) As String
	Dim f As String, p As String, i As Long, j As Long
	i = InStr(label, "f=")
	If i > 0 Then
		j = InStr(i, label, ")")
		f = Mid(label, i + 2, j - i - 2)
	End If
	i = InStr(label, "[")
	If i > 0 Then
		j = InStr(i, label, "]")
		p = Mid(label, i + 1, j - i - 1)
	End If
	If f <> "" And p <> "" Then
		FileLabel = "farfield_f" & f & "_port" & p
	Else
		FileLabel = Replace(Replace(Replace(Replace(Replace(label, " ", "_"), "(", ""), ")", ""), "[", ""), "]", "")
	End If
End Function
