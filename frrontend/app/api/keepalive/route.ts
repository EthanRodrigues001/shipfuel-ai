import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  
  try {
    // Ping the backend health route
    const res = await fetch(`${apiUrl}/health`);
    
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json({
        message: "Successfully pinged the backend",
        backend_status: data,
        success: true
      });
    } else {
      return NextResponse.json({
        message: "Backend returned an error",
        status: res.status,
        success: false
      }, { status: 502 });
    }
  } catch (error) {
    return NextResponse.json({
      message: "Failed to ping backend (might be waking up from sleep)",
      error: String(error),
      success: false
    }, { status: 504 });
  }
}
