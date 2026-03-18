"use client";

import { useState } from "react";

const MODAL_API_URL = process.env.NEXT_PUBLIC_MODAL_API_URL || "";

function ImageUpload({
  label,
  onImageSelect,
  preview,
  hint,
}: {
  label: string;
  onImageSelect: (b64: string) => void;
  preview: string | null;
  hint: string;
}) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const b64 = (reader.result as string).split(",")[1];
      onImageSelect(b64);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm font-semibold text-gray-700">{label}</p>
      <label className="border-2 border-dashed border-gray-300 rounded-xl cursor-pointer hover:border-purple-400 hover:bg-gray-50 transition-all block">
        <input type="file" accept="image/*" onChange={handleChange} className="hidden" />
        {preview ? (
          <div className="relative w-full aspect-[3/4] rounded-xl overflow-hidden">
            <img
              src={"data:image/png;base64," + preview}
              alt={label}
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-black/20 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
              <p className="text-white text-sm font-medium">Click to change</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-10 gap-3">
            <div className="w-12 h-12 rounded-full bg-purple-100 flex items-center justify-center">
              <svg className="w-6 h-6 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-gray-700">Click to upload</p>
              <p className="text-xs text-gray-400 mt-1">{hint}</p>
            </div>
          </div>
        )}
      </label>
    </div>
  );
}

export default function Home() {
  const [personB64, setPersonB64] = useState<string | null>(null);
  const [clothB64, setClothB64] = useState<string | null>(null);
  const [resultB64, setResultB64] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState("");

  const handleTryOn = async () => {
    if (!personB64 || !clothB64) {
      setError("Please upload both a person image and a clothing image.");
      return;
    }
    setLoading(true);
    setError(null);
    setResultB64(null);

    const steps = [
      "Preprocessing images...",
      "Running pose estimation...",
      "Generating segmentation...",
      "Warping garment...",
      "Synthesizing final image...",
    ];
    let stepIdx = 0;
    setProgress(steps[0]);
    const interval = setInterval(() => {
      stepIdx = (stepIdx + 1) % steps.length;
      setProgress(steps[stepIdx]);
    }, 2500);

    try {
      const response = await fetch(MODAL_API_URL + "/tryon", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          person_image: personB64,
          cloth_image: clothB64,
        }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Server error");
      }
      const data = await response.json();
      if (data.success) {
        setResultB64(data.result_image);
      } else {
        setError("Try-on failed. Please try again.");
      }
    } catch (err: any) {
      setError(err?.message || "Something went wrong.");
    } finally {
      clearInterval(interval);
      setLoading(false);
      setProgress("");
    }
  };

  const reset = () => {
    setPersonB64(null);
    setClothB64(null);
    setResultB64(null);
    setError(null);
  };

  const downloadResult = () => {
    if (!resultB64) return;
    const link = document.createElement("a");
    link.href = "data:image/png;base64," + resultB64;
    link.download = "virtualfit-result.png";
    link.click();
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-pink-50">
      <header className="border-b border-gray-100 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <span className="text-white text-sm font-bold">V</span>
            </div>
            <h1 className="text-xl font-bold text-gray-900">VirtualFit</h1>
          </div>
          <span className="text-xs text-gray-400 bg-gray-100 px-3 py-1 rounded-full">
            Powered by VITON-HD
          </span>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Try On Any Outfit,{" "}
            <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              Instantly
            </span>
          </h2>
          <p className="text-lg text-gray-500 max-w-xl mx-auto">
            Upload your photo and a clothing item. Our AI will show you exactly how it looks on you.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-6">
            <ImageUpload
              label="Your Photo"
              onImageSelect={(b64) => setPersonB64(b64)}
              preview={personB64}
              hint="Full body photo works best"
            />
            <ImageUpload
              label="Clothing Item"
              onImageSelect={(b64) => setClothB64(b64)}
              preview={clothB64}
              hint="Flat-lay or product image"
            />

            <div className="sm:col-span-2 flex flex-col gap-3">
              <button
                onClick={handleTryOn}
                disabled={!personB64 || !clothB64 || loading}
                className={
                  "w-full py-4 rounded-xl font-semibold text-white text-lg transition-all " +
                  (!personB64 || !clothB64 || loading
                    ? "bg-gray-300 cursor-not-allowed"
                    : "bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 shadow-lg")
                }
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-3">
                    <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    {progress}
                  </span>
                ) : (
                  "Try It On"
                )}
              </button>

              {(personB64 || clothB64 || resultB64) && (
                <button
                  onClick={reset}
                  className="w-full py-3 rounded-xl font-medium text-gray-600 border border-gray-200 hover:bg-gray-50 transition-all"
                >
                  Reset
                </button>
              )}
            </div>

            {error && (
              <div className="sm:col-span-2 p-4 bg-red-50 border border-red-200 rounded-xl">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}
          </div>

          <div className="flex flex-col gap-3">
            <p className="text-sm font-semibold text-gray-700">Result</p>
            <div className={"border-2 rounded-xl overflow-hidden transition-all " + (resultB64 ? "border-purple-400 shadow-xl" : "border-gray-200")}>
              {resultB64 ? (
                <div className="relative">
                  <img
                    src={"data:image/png;base64," + resultB64}
                    alt="Try-on result"
                    className="w-full"
                  />
                  <button
                    onClick={downloadResult}
                    className="absolute bottom-3 right-3 bg-white text-purple-700 text-xs font-semibold px-3 py-2 rounded-lg shadow hover:bg-purple-50 transition-all"
                  >
                    Download
                  </button>
                </div>
              ) : (
                <div className="aspect-[3/4] flex flex-col items-center justify-center gap-3 bg-gray-50">
                  {loading ? (
                    <div className="flex flex-col items-center gap-4">
                      <div className="w-12 h-12 rounded-full border-4 border-purple-200 border-t-purple-600 animate-spin" />
                      <p className="text-sm text-gray-500 text-center px-4">{progress}</p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center gap-3">
                      <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center">
                        <span className="text-3xl">👗</span>
                      </div>
                      <p className="text-sm text-gray-400 text-center px-6">
                        Your virtual try-on result will appear here
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-20">
          <h3 className="text-2xl font-bold text-center text-gray-800 mb-8">How It Works</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            {[
              { icon: "📸", title: "Upload Your Photo", desc: "A full-body photo with clear visibility of your torso works best" },
              { icon: "👕", title: "Pick a Garment", desc: "Upload any clothing item on a white background" },
              { icon: "✨", title: "See the Result", desc: "Our AI places the garment on your body using VITON-HD" },
            ].map(({ icon, title, desc }) => (
              <div key={title} className="flex flex-col items-center text-center p-6 rounded-2xl bg-white border border-gray-100 shadow-sm">
                <div className="w-12 h-12 rounded-full bg-purple-100 flex items-center justify-center text-2xl mb-4">
                  {icon}
                </div>
                <h4 className="font-semibold text-gray-800 mb-2">{title}</h4>
                <p className="text-sm text-gray-500">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}